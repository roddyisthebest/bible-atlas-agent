"""FastAPI service exposing agents.places.workflow.graph.

Endpoints:
    GET  /health   — liveness (no auth)
    POST /invoke   — sync agent call (X-API-Key)
    POST /stream   — SSE stream of graph updates (X-API-Key)
"""
from __future__ import annotations

import logging
import os
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.load import dumps as lc_dumps
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from agents.places.workflow import graph

load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI(title="bible-atlas-agent")


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


def require_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    expected = os.environ.get("API_KEY")
    if not expected or x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid api key")


class InvokeRequest(BaseModel):
    query: str = Field(min_length=1)


class InvokeResponse(BaseModel):
    answer: str
    place_id_map: dict[str, list[str]] = Field(default_factory=dict)
    recommended_questions: list[str] = Field(default_factory=list)


def _sse_event(event: str, data: object) -> str:
    """Serialize a single SSE event. LangChain objects survive via lc_dumps."""
    payload = lc_dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


def _stream_graph(query: str):
    initial_state = {
        "query": query,
        "messages": [HumanMessage(content=query)],
    }
    final: dict = {}
    try:
        for update in graph.stream(initial_state, stream_mode="updates"):
            # update is {node_name: delta}. delta may include LangChain messages.
            (node_name, delta), = update.items()
            yield _sse_event("node", {"node": node_name, "update": delta})
            if delta:
                for key in ("answer", "place_id_map", "recommended_questions"):
                    if key in delta:
                        final[key] = delta[key]
    except Exception as exc:  # noqa: BLE001 — user-facing error stream
        logger.exception("graph.stream failed")
        yield _sse_event("error", {"detail": str(exc)})
        return

    done_payload = {
        "answer": final.get("answer", ""),
        "place_id_map": final.get("place_id_map") or {},
        "recommended_questions": final.get("recommended_questions") or [],
    }
    yield _sse_event("done", done_payload)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post(
    "/invoke",
    dependencies=[Depends(require_api_key)],
    response_model=InvokeResponse,
)
def invoke(request: InvokeRequest) -> InvokeResponse:
    initial_state = {
        "query": request.query,
        "messages": [HumanMessage(content=request.query)],
    }
    result = graph.invoke(initial_state)
    return InvokeResponse(
        answer=result.get("answer", ""),
        place_id_map=result.get("place_id_map") or {},
        recommended_questions=result.get("recommended_questions") or [],
    )


@app.post("/stream", dependencies=[Depends(require_api_key)])
def stream(request: InvokeRequest) -> StreamingResponse:
    return StreamingResponse(
        _stream_graph(request.query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
