"""FastAPI service exposing agents.places.workflow.graph.

Endpoints:
    GET  /health   — liveness (no auth)
    POST /invoke   — sync agent call (X-API-Key)
    POST /stream   — SSE stream of graph updates (X-API-Key)
"""
from __future__ import annotations

import logging
import os
from typing import Annotated, Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.load import dumps as lc_dumps
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field

from agents.places import chat as chat_orchestration
from agents.places.chat import ChatMessage as _ChatMessage
from agents.places.workflow import graph

load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI(title="bible-atlas-agent")

# TODO: restrict for prod. Local dev opens everything.
# X-API-Key auth uses a custom request header, not cookies, so
# allow_credentials is left off — this lets the middleware return
# "*" as the ACAO header (Fetch spec forbids "*" with credentials).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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


class ChatMessagePayload(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class InvokeRequest(BaseModel):
    query: str = Field(min_length=1)
    summary: str | None = None
    messages: list[ChatMessagePayload] = Field(default_factory=list)


class InvokeResponse(BaseModel):
    answer: str
    place_id_map: dict[str, list[str]] = Field(default_factory=dict)
    recommended_questions: list[str] = Field(default_factory=list)
    summary: str | None = None
    messages: list[ChatMessagePayload] = Field(default_factory=list)


def _to_chat_messages(payload: list[ChatMessagePayload]) -> list[_ChatMessage]:
    return [_ChatMessage(role=m.role, content=m.content) for m in payload]


def _to_payload(msgs: list[_ChatMessage]) -> list[ChatMessagePayload]:
    return [ChatMessagePayload(role=m.role, content=m.content) for m in msgs]


def _sse_event(event: str, data: object) -> str:
    """Serialize a single SSE event. LangChain objects survive via lc_dumps."""
    payload = lc_dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


def _stream_graph(
    query: str,
    summary: str | None,
    messages_payload: list[ChatMessagePayload],
):
    prior = _to_chat_messages(messages_payload)
    ctx_messages = chat_orchestration.build_context(
        summary=summary, messages=prior, query=query,
    )
    initial_state = {"query": query, "messages": ctx_messages}
    final: dict = {}
    # place_agent 는 create_agent 의 ReAct tool loop 를 서브그래프로 돌리기
    # 때문에 outer stream_mode="updates" 만으론 그 사이 침묵이 길다.
    # subgraphs=True + ["updates","messages"] 로 서브그래프 내부 이벤트까지
    # 관통시켜, 도구 호출/완료마다 실시간 `tool` 이벤트를 흘린다.
    #
    # subgraphs=True 시 yield 시그니처: (namespace, mode, chunk).
    #   namespace = () 는 outer graph, non-empty tuple 은 subgraph 노드.
    seen_tool_starts: set[str] = set()
    seen_tool_dones: set[str] = set()
    try:
        for namespace, mode, chunk in graph.stream(
            initial_state,
            stream_mode=["updates", "messages"],
            subgraphs=True,
        ):
            if mode == "messages":
                # chunk = (message, metadata); metadata unused here.
                msg = chunk[0]
                # 도구 호출 시작: AIMessage(Chunk) 의 tool_calls 목록에서
                # 처음 보는 id 마다 한 번씩 `tool` 이벤트 발화.
                for tc in getattr(msg, "tool_calls", None) or []:
                    tc_id = tc.get("id")
                    if tc_id and tc_id not in seen_tool_starts:
                        seen_tool_starts.add(tc_id)
                        yield _sse_event("tool", {
                            "id": tc_id,
                            "name": tc.get("name"),
                            "phase": "start",
                        })
                # 도구 완료: ToolMessage 도착 시 한 번.
                if isinstance(msg, ToolMessage):
                    tc_id = getattr(msg, "tool_call_id", None)
                    if tc_id and tc_id not in seen_tool_dones:
                        seen_tool_dones.add(tc_id)
                        yield _sse_event("tool", {
                            "id": tc_id,
                            "name": msg.name,
                            "phase": "done",
                        })
                continue

            # mode == "updates" — outer graph 의 노드 완료 이벤트만 전송.
            # subgraph 노드 업데이트(namespace != ()) 는 노이즈라 스킵.
            if namespace:
                continue
            for node_name, delta in chunk.items():
                yield _sse_event("node", {"node": node_name, "update": delta})
                if delta:
                    for key in ("answer", "place_id_map", "recommended_questions"):
                        if key in delta:
                            final[key] = delta[key]
    except Exception as exc:  # noqa: BLE001 — user-facing error stream
        logger.exception("graph.stream failed")
        yield _sse_event("error", {"detail": str(exc)})
        return

    turn = chat_orchestration.finalize_turn(
        query=query,
        summary=summary,
        messages=prior,
        answer=final.get("answer", ""),
        place_id_map=final.get("place_id_map"),
        recommended_questions=final.get("recommended_questions"),
    )
    done_payload = {
        "answer": turn.answer,
        "place_id_map": turn.place_id_map,
        "recommended_questions": turn.recommended_questions,
        "summary": turn.summary,
        "messages": [{"role": m.role, "content": m.content} for m in turn.messages],
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
    turn = chat_orchestration.run_turn(
        query=request.query,
        summary=request.summary,
        messages=_to_chat_messages(request.messages),
        graph=graph,
    )
    return InvokeResponse(
        answer=turn.answer,
        place_id_map=turn.place_id_map,
        recommended_questions=turn.recommended_questions,
        summary=turn.summary,
        messages=_to_payload(turn.messages),
    )


# Note: if the client disconnects mid-stream, the in-flight graph.stream()
# step continues to completion in the threadpool (sync generator limitation).
# The generator is then abandoned — no leaks, but the current LLM/DB call
# is not cancelled. Revisit if we move to an async graph runtime.
@app.post("/stream", dependencies=[Depends(require_api_key)])
def stream(request: InvokeRequest) -> StreamingResponse:
    return StreamingResponse(
        _stream_graph(request.query, request.summary, request.messages),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
