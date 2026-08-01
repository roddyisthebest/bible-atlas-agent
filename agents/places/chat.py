"""Chat orchestration: multi-turn context + summary management.

Wraps `agents.places.workflow.graph` with a stateless per-turn interface
the HTTP layer can call. Client keeps `summary` + `messages`; server
rebuilds context from them on each turn.

See docs/chat-api-spec.md for the wire contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


THRESHOLD = 10  # messages count (user+assistant combined) that triggers summary
SUMMARY_MAX_CHARS = 1600  # hard truncate defense
SUMMARY_MAX_TOKENS = 400  # LLM max_tokens


@dataclass
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass
class ChatTurnResult:
    answer: str
    place_id_map: dict[str, list[str]]
    recommended_questions: list[str]
    summary: str | None
    messages: list["ChatMessage"]


def should_summarize(messages: list[ChatMessage]) -> bool:
    return len(messages) > THRESHOLD


def build_context(
    *,
    summary: str | None,
    messages: list[ChatMessage],
    query: str,
) -> list[BaseMessage]:
    """Assemble LangChain message list for a single graph invocation."""
    ctx: list[BaseMessage] = []
    if summary:
        ctx.append(SystemMessage(content=f"이전 대화 요약:\n{summary}"))
    for m in messages:
        if m.role == "user":
            ctx.append(HumanMessage(content=m.content))
        else:
            ctx.append(AIMessage(content=m.content))
    ctx.append(HumanMessage(content=query))
    return ctx


def summarize(
    *,
    prev_summary: str | None,
    messages: list[ChatMessage],
    chain: Callable[[dict], str] | None = None,
) -> str:
    """Compress (prev_summary + messages) into a bounded summary string.

    `chain` is any callable taking a dict input and returning a string.
    In production this is a LangChain prompt|llm|parser pipeline (built lazily
    on first use). Tests inject a fake to avoid real LLM calls.
    """
    if chain is None:
        chain = _default_summary_chain()
    raw = chain({
        "prev_summary": prev_summary or "",
        "messages": _format_messages_for_summary(messages),
    })
    return raw[:SUMMARY_MAX_CHARS]


def _format_messages_for_summary(messages: list[ChatMessage]) -> str:
    return "\n".join(f"{m.role}: {m.content}" for m in messages)


def finalize_turn(
    *,
    query: str,
    summary: str | None,
    messages: list[ChatMessage],
    answer: str,
    place_id_map: dict[str, list[str]] | None = None,
    recommended_questions: list[str] | None = None,
    summarize_fn: Callable[..., str] | None = None,
) -> ChatTurnResult:
    """Post-graph bookkeeping shared by run_turn (graph.invoke) and streaming
    callers that drive graph.stream themselves.

    Appends (user query, assistant answer) to messages, and summarizes+resets
    when the running count exceeds THRESHOLD.
    """
    if summarize_fn is None:
        summarize_fn = summarize

    new_messages = list(messages) + [
        ChatMessage(role="user", content=query),
        ChatMessage(role="assistant", content=answer),
    ]
    new_summary = summary
    if should_summarize(new_messages):
        new_summary = summarize_fn(prev_summary=summary, messages=new_messages)
        new_messages = []

    return ChatTurnResult(
        answer=answer,
        place_id_map=place_id_map or {},
        recommended_questions=recommended_questions or [],
        summary=new_summary,
        messages=new_messages,
    )


def run_turn(
    query: str,
    summary: str | None = None,
    messages: list[ChatMessage] | None = None,
    *,
    graph=None,
    summarize_fn: Callable[..., str] | None = None,
) -> ChatTurnResult:
    """Run one chat turn: build context → invoke graph → finalize.

    `graph` and `summarize_fn` are injectable for tests. Defaults resolve
    lazily to the real workflow graph and the module-level `summarize`.
    """
    messages = messages or []
    if graph is None:
        from agents.places.workflow import graph as default_graph
        graph = default_graph

    ctx_messages = build_context(summary=summary, messages=messages, query=query)
    result = graph.invoke({"query": query, "messages": ctx_messages})

    return finalize_turn(
        query=query,
        summary=summary,
        messages=messages,
        answer=result.get("answer", ""),
        place_id_map=result.get("place_id_map"),
        recommended_questions=result.get("recommended_questions"),
        summarize_fn=summarize_fn,
    )


_summary_chain_singleton = None


def _default_summary_chain():
    global _summary_chain_singleton
    if _summary_chain_singleton is None:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=SUMMARY_MAX_TOKENS,
        )
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "당신은 성경 지리 챗봇의 대화 요약자입니다.\n"
                "아래 [이전 요약]과 [최근 대화]를 합쳐 하나의 새 요약을 만드세요.\n"
                "규칙:\n"
                "- 400 토큰(한글 약 800자) 이내로 압축.\n"
                "- 사용자의 주제·관심사, 지금까지 언급된 지명, 대화 흐름 요지만 남길 것.\n"
                "- 이모지·인사말·메타 발언 금지. 서술식 짧은 문단.\n",
            ),
            (
                "user",
                "[이전 요약]\n{prev_summary}\n\n[최근 대화]\n{messages}\n\n[새 요약]",
            ),
        ])
        _summary_chain_singleton = (prompt | llm | StrOutputParser()).invoke
    return _summary_chain_singleton
