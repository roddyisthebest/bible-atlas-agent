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
    assert body == {
        "answer": "just an answer",
        "place_id_map": {},
        "recommended_questions": [],
    }


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
    # When allow_credentials=True, Starlette reflects the request origin instead
    # of "*" (per the CORS spec, credentials + wildcard is disallowed).
    acao = response.headers.get("access-control-allow-origin")
    assert acao in ("*", "http://localhost:3000")


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
