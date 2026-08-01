"""Tests for agents.places.chat — chat orchestration (multi-turn + summary).

Graph and summarize LLM are injected so no real LLM/DB calls happen.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from types import SimpleNamespace

from agents.places.chat import (
    SUMMARY_MAX_CHARS,
    THRESHOLD,
    ChatMessage,
    build_context,
    finalize_turn,
    run_turn,
    should_summarize,
    summarize,
)


def _fake_graph(invoke_result: dict):
    captured = {}

    def _invoke(state):
        captured["state"] = state
        return invoke_result

    return SimpleNamespace(invoke=_invoke), captured


def test_should_summarize_below_threshold_returns_false():
    messages = [ChatMessage(role="user", content=f"m{i}") for i in range(THRESHOLD - 2)]
    assert should_summarize(messages) is False


def test_should_summarize_at_threshold_returns_false():
    # Spec: summarize only when count EXCEEDS threshold (strictly greater).
    messages = [ChatMessage(role="user", content=f"m{i}") for i in range(THRESHOLD)]
    assert should_summarize(messages) is False


def test_should_summarize_above_threshold_returns_true():
    messages = [ChatMessage(role="user", content=f"m{i}") for i in range(THRESHOLD + 1)]
    assert should_summarize(messages) is True


def test_build_context_first_turn_has_only_query():
    ctx = build_context(summary=None, messages=[], query="베들레헴 알려줘")
    assert len(ctx) == 1
    assert isinstance(ctx[0], HumanMessage)
    assert ctx[0].content == "베들레헴 알려줘"


def test_build_context_prepends_summary_as_system_message():
    ctx = build_context(summary="이전 대화 요지", messages=[], query="다음은?")
    assert len(ctx) == 2
    assert isinstance(ctx[0], SystemMessage)
    assert "이전 대화 요지" in ctx[0].content
    assert isinstance(ctx[1], HumanMessage)
    assert ctx[1].content == "다음은?"


def test_build_context_converts_prior_messages_by_role():
    prior = [
        ChatMessage(role="user", content="q1"),
        ChatMessage(role="assistant", content="a1"),
        ChatMessage(role="user", content="q2"),
        ChatMessage(role="assistant", content="a2"),
    ]
    ctx = build_context(summary=None, messages=prior, query="q3")
    assert [type(m) for m in ctx] == [
        HumanMessage, AIMessage, HumanMessage, AIMessage, HumanMessage,
    ]
    assert [m.content for m in ctx] == ["q1", "a1", "q2", "a2", "q3"]


def test_build_context_summary_then_prior_then_query_order():
    prior = [ChatMessage(role="user", content="q1"), ChatMessage(role="assistant", content="a1")]
    ctx = build_context(summary="요약본", messages=prior, query="q2")
    assert isinstance(ctx[0], SystemMessage)
    assert ctx[-1].content == "q2"
    assert len(ctx) == 4  # system + 2 prior + query


def test_summarize_returns_chain_output_when_under_limit():
    fake_chain = lambda _inputs: "짧은 요약본"
    result = summarize(prev_summary=None, messages=[], chain=fake_chain)
    assert result == "짧은 요약본"


def test_summarize_hard_truncates_over_limit():
    # Defense in depth: even if LLM ignores max_tokens, wire never gets more
    # than SUMMARY_MAX_CHARS.
    oversize = "가" * (SUMMARY_MAX_CHARS + 500)
    result = summarize(
        prev_summary=None,
        messages=[],
        chain=lambda _inputs: oversize,
    )
    assert len(result) == SUMMARY_MAX_CHARS


def test_summarize_passes_prev_and_messages_to_chain():
    captured = {}

    def fake_chain(inputs):
        captured.update(inputs)
        return "ok"

    summarize(
        prev_summary="지난 요약",
        messages=[
            ChatMessage(role="user", content="q1"),
            ChatMessage(role="assistant", content="a1"),
        ],
        chain=fake_chain,
    )
    assert captured["prev_summary"] == "지난 요약"
    assert "user: q1" in captured["messages"]
    assert "assistant: a1" in captured["messages"]


def test_summarize_none_prev_summary_becomes_empty_string_for_chain():
    captured = {}

    def fake_chain(inputs):
        captured.update(inputs)
        return "ok"

    summarize(prev_summary=None, messages=[], chain=fake_chain)
    assert captured["prev_summary"] == ""


def test_run_turn_first_turn_appends_pair_and_no_summary():
    graph, _ = _fake_graph({
        "answer": "베들레헴은…",
        "place_id_map": {"베들레헴": ["a112427"]},
        "recommended_questions": [],
    })
    result = run_turn(
        query="베들레헴 알려줘",
        summary=None,
        messages=None,
        graph=graph,
    )
    assert result.answer == "베들레헴은…"
    assert result.place_id_map == {"베들레헴": ["a112427"]}
    assert result.recommended_questions == []
    assert result.summary is None
    assert result.messages == [
        ChatMessage(role="user", content="베들레헴 알려줘"),
        ChatMessage(role="assistant", content="베들레헴은…"),
    ]


def test_run_turn_below_threshold_preserves_summary_and_appends():
    graph, _ = _fake_graph({"answer": "a2"})
    prior = [
        ChatMessage(role="user", content="q1"),
        ChatMessage(role="assistant", content="a1"),
    ]
    result = run_turn(
        query="q2",
        summary="이전 요약",
        messages=prior,
        graph=graph,
    )
    assert result.summary == "이전 요약"  # untouched
    assert result.messages == prior + [
        ChatMessage(role="user", content="q2"),
        ChatMessage(role="assistant", content="a2"),
    ]


def test_run_turn_triggers_summary_when_exceeding_threshold():
    graph, _ = _fake_graph({"answer": "final"})
    # Prior 10 messages + this turn's 2 = 12 → exceeds THRESHOLD (10).
    prior = [
        ChatMessage(role="user" if i % 2 == 0 else "assistant", content=f"m{i}")
        for i in range(THRESHOLD)
    ]

    summarize_calls = []

    def fake_summarize_fn(*, prev_summary, messages):
        summarize_calls.append({"prev_summary": prev_summary, "messages": list(messages)})
        return "새 요약"

    result = run_turn(
        query="last q",
        summary="옛 요약",
        messages=prior,
        graph=graph,
        summarize_fn=fake_summarize_fn,
    )
    assert result.summary == "새 요약"
    assert result.messages == []  # cleared after summarization
    assert len(summarize_calls) == 1
    assert summarize_calls[0]["prev_summary"] == "옛 요약"
    # summarize should see the messages including THIS turn's pair
    assert summarize_calls[0]["messages"][-1] == ChatMessage(role="assistant", content="final")


def test_run_turn_passes_built_context_to_graph():
    graph, captured = _fake_graph({"answer": "a2"})
    prior = [
        ChatMessage(role="user", content="q1"),
        ChatMessage(role="assistant", content="a1"),
    ]
    run_turn(query="q2", summary="요약", messages=prior, graph=graph)

    state = captured["state"]
    assert state["query"] == "q2"
    # SystemMessage(summary) + HumanMessage(q1) + AIMessage(a1) + HumanMessage(q2)
    assert len(state["messages"]) == 4
    from langchain_core.messages import HumanMessage as HM
    from langchain_core.messages import SystemMessage as SM
    assert isinstance(state["messages"][0], SM)
    assert state["messages"][-1].content == "q2"
    assert isinstance(state["messages"][-1], HM)


def test_run_turn_defaults_missing_optional_fields_to_empty():
    # Graph only returns "answer" (bible_general_agent / rewrite path).
    graph, _ = _fake_graph({"answer": "just answer"})
    result = run_turn(query="q", graph=graph)
    assert result.place_id_map == {}
    assert result.recommended_questions == []


# ---------------------------------------------------------------------------
# finalize_turn: shared post-graph bookkeeping (used by run_turn + /stream)
# ---------------------------------------------------------------------------
def test_finalize_turn_appends_user_and_assistant_pair():
    prior = [ChatMessage(role="user", content="q1"), ChatMessage(role="assistant", content="a1")]
    result = finalize_turn(
        query="q2",
        summary=None,
        messages=prior,
        answer="a2",
        place_id_map={"베들레헴": ["a1"]},
        recommended_questions=[],
    )
    assert result.answer == "a2"
    assert result.summary is None
    assert result.messages == [
        ChatMessage(role="user", content="q1"),
        ChatMessage(role="assistant", content="a1"),
        ChatMessage(role="user", content="q2"),
        ChatMessage(role="assistant", content="a2"),
    ]
    assert result.place_id_map == {"베들레헴": ["a1"]}


def test_finalize_turn_defaults_missing_optionals():
    result = finalize_turn(
        query="q", summary=None, messages=[], answer="a",
    )
    assert result.place_id_map == {}
    assert result.recommended_questions == []


def test_finalize_turn_triggers_summarize_and_resets_messages():
    # Fill just below threshold so the (user, assistant) pair pushes over.
    prior = [
        ChatMessage(role="user" if i % 2 == 0 else "assistant", content=f"m{i}")
        for i in range(THRESHOLD - 1)
    ]
    summarize_calls: list[dict] = []

    def fake_summarize(*, prev_summary, messages):
        summarize_calls.append({"prev_summary": prev_summary, "messages": messages})
        return "요약본"

    result = finalize_turn(
        query="q_last",
        summary="옛 요약",
        messages=prior,
        answer="a_last",
        summarize_fn=fake_summarize,
    )
    assert result.summary == "요약본"
    assert result.messages == []
    assert len(summarize_calls) == 1
    assert summarize_calls[0]["prev_summary"] == "옛 요약"
    assert summarize_calls[0]["messages"][-1] == ChatMessage(role="assistant", content="a_last")
