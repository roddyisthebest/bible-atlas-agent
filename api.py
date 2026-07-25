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
from fastapi.responses import JSONResponse
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
