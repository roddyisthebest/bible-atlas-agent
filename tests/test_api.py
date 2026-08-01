"""Tests for the FastAPI service in api.py.

Graph is mocked so no real LLM/DB calls happen. Requires local .env
loaded so that api module import (which pulls agents.places.workflow)
does not crash on missing POSTGRES_URL / OPENAI_API_KEY.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    # Ensure API_KEY is set BEFORE importing api so require_api_key comparison works.
    os.environ.setdefault("API_KEY", "test-key")
    from api import app
    # raise_server_exceptions=False lets the FastAPI exception handler return a
    # proper JSON 500 response instead of having TestClient re-raise the error.
    return TestClient(app, raise_server_exceptions=False)


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


@pytest.fixture
def api_headers():
    return {"X-API-Key": os.environ["API_KEY"]}


def test_invoke_without_key_returns_401(client):
    response = client.post("/invoke", json={"query": "hi"})
    assert response.status_code == 401
    assert response.json() == {"detail": "invalid api key"}


def test_invoke_with_wrong_key_returns_401(client):
    response = client.post(
        "/invoke",
        json={"query": "hi"},
        headers={"X-API-Key": "nope"},
    )
    assert response.status_code == 401


def test_invoke_with_valid_key_reaches_handler(client, api_headers, fake_graph):
    # Prove auth passes and the handler reaches graph.invoke. Response shape
    # is separately tested by test_invoke_returns_response_shape.
    fake_graph(invoke_result={"answer": "ok"})
    response = client.post("/invoke", json={"query": "hi"}, headers=api_headers)
    assert response.status_code == 200


@pytest.fixture
def fake_graph(monkeypatch):
    """Replace api.graph with a controllable fake."""
    def _install(*, invoke_result=None, stream_updates=None):
        fake = SimpleNamespace(
            invoke=lambda state: invoke_result or {},
            stream=lambda state, stream_mode: iter(stream_updates or []),
        )
        monkeypatch.setattr("api.graph", fake)
        return fake

    return _install


def test_invoke_returns_response_shape(client, api_headers, fake_graph):
    fake_graph(invoke_result={
        "answer": "베들레헴은 유다 지파의 성읍.",
        "place_id_map": {"베들레헴": ["a112427"]},
        "recommended_questions": [],
    })

    response = client.post(
        "/invoke",
        json={"query": "베들레헴에 대해 알려줘"},
        headers=api_headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("베들레헴")
    assert body["place_id_map"] == {"베들레헴": ["a112427"]}
    assert body["recommended_questions"] == []


def test_invoke_missing_optional_fields_defaults_to_empty(client, api_headers, fake_graph):
    # graph.invoke may return only "answer" (e.g. bible_general_agent path).
    fake_graph(invoke_result={"answer": "just an answer"})

    response = client.post(
        "/invoke",
        json={"query": "성령이란?"},
        headers=api_headers,
    )
    assert response.status_code == 200
    body = response.json()
    # First turn: server echoes (user, assistant) pair in messages, summary null.
    assert body == {
        "answer": "just an answer",
        "place_id_map": {},
        "recommended_questions": [],
        "summary": None,
        "messages": [
            {"role": "user", "content": "성령이란?"},
            {"role": "assistant", "content": "just an answer"},
        ],
    }


def test_invoke_forwards_prior_summary_and_messages_into_graph_context(
    client, api_headers, fake_graph
):
    fake = fake_graph(invoke_result={"answer": "a2"})
    captured: dict = {}
    fake.invoke = lambda state: (captured.update({"state": state}) or {"answer": "a2"})
    # fake_graph fixture sets api.graph; overwrite the invoke to capture.
    import api
    api.graph = fake

    response = client.post(
        "/invoke",
        json={
            "query": "다음은?",
            "summary": "지난 대화 요지",
            "messages": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
            ],
        },
        headers=api_headers,
    )
    assert response.status_code == 200

    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    ctx = captured["state"]["messages"]
    # SystemMessage(summary) + HumanMessage(q1) + AIMessage(a1) + HumanMessage(다음은?)
    assert [type(m) for m in ctx] == [SystemMessage, HumanMessage, AIMessage, HumanMessage]
    assert "지난 대화 요지" in ctx[0].content
    assert ctx[-1].content == "다음은?"


def test_invoke_returns_appended_messages_carrying_prior_history(
    client, api_headers, fake_graph
):
    fake_graph(invoke_result={"answer": "a2"})
    response = client.post(
        "/invoke",
        json={
            "query": "q2",
            "messages": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
            ],
        },
        headers=api_headers,
    )
    body = response.json()
    assert body["messages"] == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    assert body["summary"] is None


def test_invoke_empty_query_returns_422(client, api_headers):
    response = client.post("/invoke", json={"query": ""}, headers=api_headers)
    assert response.status_code == 422


def test_invoke_missing_body_returns_422(client, api_headers):
    response = client.post("/invoke", headers=api_headers)
    assert response.status_code == 422


def test_invoke_graph_exception_returns_500_with_detail(client, api_headers, monkeypatch):
    def boom(state):
        raise RuntimeError("db offline")

    monkeypatch.setattr(
        "api.graph",
        SimpleNamespace(invoke=boom, stream=lambda s, stream_mode: iter([])),
    )

    response = client.post(
        "/invoke",
        json={"query": "hi"},
        headers=api_headers,
    )
    assert response.status_code == 500
    assert response.json() == {"detail": "db offline"}


def _parse_sse(body: str) -> list[dict]:
    """Minimal SSE parser: split on blank lines, extract event + data."""
    import json
    events = []
    for chunk in body.strip().split("\n\n"):
        event_type = None
        data = None
        for line in chunk.splitlines():
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = json.loads(line[len("data:"):].strip())
        events.append({"event": event_type, "data": data})
    return events


def test_stream_without_key_returns_401(client):
    response = client.post("/stream", json={"query": "hi"})
    assert response.status_code == 401


def test_stream_emits_node_events_then_done(client, api_headers, fake_graph):
    fake_graph(stream_updates=[
        {"router": None},
        {"place_agent": {"messages": []}},
        {"format": {"answer": "hi", "place_id_map": {"베들레헴": ["a112427"]}}},
    ])

    with client.stream(
        "POST",
        "/stream",
        json={"query": "hi"},
        headers=api_headers,
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    events = _parse_sse(body)
    # 3 node events + 1 done event
    assert [e["event"] for e in events] == ["node", "node", "node", "done"]
    assert events[0]["data"] == {"node": "router", "update": None}
    assert events[-1]["data"]["answer"] == "hi"
    assert events[-1]["data"]["place_id_map"] == {"베들레헴": ["a112427"]}
    assert events[-1]["data"]["recommended_questions"] == []


def test_cors_preflight_allows_frontend(client):
    response = client.options(
        "/invoke",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "X-API-Key,Content-Type",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "*"


def test_stream_emits_error_event_on_graph_exception(client, api_headers, monkeypatch):
    def boom(state, stream_mode):
        yield {"router": None}
        raise RuntimeError("kaboom")

    monkeypatch.setattr(
        "api.graph",
        SimpleNamespace(invoke=lambda s: {}, stream=boom),
    )

    with client.stream(
        "POST",
        "/stream",
        json={"query": "hi"},
        headers=api_headers,
    ) as response:
        body = "".join(response.iter_text())

    events = _parse_sse(body)
    assert events[0]["event"] == "node"
    assert events[-1]["event"] == "error"
    assert "kaboom" in events[-1]["data"]["detail"]


def test_stream_done_payload_includes_summary_and_messages(client, api_headers, fake_graph):
    fake_graph(stream_updates=[
        {"format": {"answer": "hi", "place_id_map": {}}},
    ])

    with client.stream(
        "POST", "/stream", json={"query": "hi"}, headers=api_headers,
    ) as response:
        body = "".join(response.iter_text())

    events = _parse_sse(body)
    done = events[-1]
    assert done["event"] == "done"
    # First turn: server echoes (user, assistant) pair; summary null.
    assert done["data"]["summary"] is None
    assert done["data"]["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hi"},
    ]


def test_stream_forwards_prior_summary_and_messages_into_graph_context(
    client, api_headers, monkeypatch
):
    captured: dict = {}

    def _stream(state, stream_mode):
        captured["state"] = state
        yield {"format": {"answer": "a2"}}

    monkeypatch.setattr(
        "api.graph",
        SimpleNamespace(invoke=lambda s: {}, stream=_stream),
    )

    with client.stream(
        "POST",
        "/stream",
        json={
            "query": "다음은?",
            "summary": "지난 대화 요지",
            "messages": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
            ],
        },
        headers=api_headers,
    ) as response:
        # Drain the stream so the generator runs to completion (finalize_turn
        # etc). Then assert on captured state.
        _ = "".join(response.iter_text())

    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    ctx = captured["state"]["messages"]
    assert [type(m) for m in ctx] == [SystemMessage, HumanMessage, AIMessage, HumanMessage]
    assert "지난 대화 요지" in ctx[0].content
    assert ctx[-1].content == "다음은?"


def test_stream_done_carries_appended_prior_messages(client, api_headers, fake_graph):
    fake_graph(stream_updates=[{"format": {"answer": "a2"}}])

    with client.stream(
        "POST",
        "/stream",
        json={
            "query": "q2",
            "messages": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
            ],
        },
        headers=api_headers,
    ) as response:
        body = "".join(response.iter_text())

    done = _parse_sse(body)[-1]
    assert done["data"]["messages"] == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    assert done["data"]["summary"] is None


def test_stream_done_triggers_summary_and_resets_messages_over_threshold(
    client, api_headers, fake_graph, monkeypatch
):
    from agents.places.chat import THRESHOLD

    fake_graph(stream_updates=[{"format": {"answer": "a_last"}}])
    monkeypatch.setattr(
        "agents.places.chat.summarize",
        lambda *, prev_summary, messages: "요약본",
    )
    # (THRESHOLD - 1) prior msgs + this turn's 2 = THRESHOLD + 1 → over threshold.
    prior = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
        for i in range(THRESHOLD - 1)
    ]

    with client.stream(
        "POST",
        "/stream",
        json={"query": "q_last", "summary": "옛 요약", "messages": prior},
        headers=api_headers,
    ) as response:
        body = "".join(response.iter_text())

    done = _parse_sse(body)[-1]
    assert done["event"] == "done"
    assert done["data"]["summary"] == "요약본"
    assert done["data"]["messages"] == []


def test_cors_header_present_on_actual_request(client):
    """Preflight OPTIONS is covered separately; verify ACAO also appears on
    real requests (regression guard for future middleware config changes)."""
    response = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "*"
