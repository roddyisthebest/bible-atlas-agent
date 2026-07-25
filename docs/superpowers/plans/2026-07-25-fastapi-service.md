# FastAPI Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose `agents.places.workflow.graph` as an HTTP service with `/health`, `/invoke` (sync), `/stream` (SSE) on `api:app`, protected by an `X-API-Key` header.

**Architecture:** Single-file `api.py` at project root. FastAPI reuses the existing compiled LangGraph graph unmodified. Each request is stateless. Tests mock the graph via `monkeypatch` so no real LLM/DB calls happen.

**Tech Stack:** FastAPI, uvicorn, Pydantic, `langchain_core.load.dumps` for SSE serialization, pytest + `TestClient` for tests.

**Design spec:** `docs/superpowers/specs/2026-07-25-fastapi-service-design.md`

**Assumed context:** local Postgres running and `.env` populated (required because `agents.places.place_tools` opens a Postgres pool at import time).

---

## File Structure

- **Create** `api.py` — FastAPI app: `app`, `require_api_key`, `InvokeRequest`, `InvokeResponse`, `/health`, `/invoke`, `/stream`, `CORSMiddleware`. Target ~150 LoC.
- **Create** `tests/test_api.py` — pytest module with fixtures (`client`, `api_headers`, `fake_graph`) and cases from the spec.
- **Modify** `pyproject.toml` — add `fastapi`, `uvicorn[standard]` deps.
- **Modify** `.env.example` — document `API_KEY` var.
- **Modify** `.env` (local, gitignored) — add real `API_KEY` value.

All handler logic, schemas, and middleware live in one file (per spec). If `api.py` grows past ~200 LoC during implementation, stop and re-scope; do not silently split.

---

## Task 1: Add dependencies and API_KEY env var

**Files:**
- Modify: `pyproject.toml`
- Modify: `.env.example`
- Modify: `.env` (local only; do NOT commit)

- [ ] **Step 1: Add FastAPI + uvicorn + httpx to `pyproject.toml`**

Edit the `dependencies = [...]` array. Insert alphabetically near existing entries:

```toml
    "fastapi>=0.115.0",
    "httpx>=0.27.0",
    ...
    "uvicorn[standard]>=0.32.0",
```

(`httpx` is required by FastAPI's `TestClient` and is NOT a transitive dep of
`fastapi` alone — must be listed explicitly.)

- [ ] **Step 2: Sync deps**

Run: `uv sync`
Expected: resolves and installs `fastapi`, `uvicorn`, `starlette`, `httpx`, no errors.

- [ ] **Step 3: Add `API_KEY` line to `.env.example`**

Append:

```
API_KEY=change-me
```

- [ ] **Step 4: Add real `API_KEY` to local `.env`**

Generate a value and append to `.env`:

```bash
echo "API_KEY=$(openssl rand -hex 32)" >> .env
```

Verify: `grep API_KEY .env` shows a 64-hex value.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock .env.example
git commit -m "chore: add fastapi/uvicorn deps and API_KEY env var"
```

**Do not stage `.env`** (gitignored via `.env.*` / `.env` patterns).

---

## Task 2: Bootstrap `api.py` with `/health` (TDD)

**Files:**
- Create: `api.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_api.py`:

```python
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
    return TestClient(app)


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}
```

- [ ] **Step 2: Run test, verify failure**

Run: `uv run pytest tests/test_api.py::test_health_returns_ok -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api'`.

- [ ] **Step 3: Create minimal `api.py`**

```python
"""FastAPI service exposing agents.places.workflow.graph.

Endpoints:
    GET  /health   — liveness (no auth)
    POST /invoke   — sync agent call (X-API-Key)
    POST /stream   — SSE stream of graph updates (X-API-Key)
"""
from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

app = FastAPI(title="bible-atlas-agent")


@app.get("/health")
def health() -> dict:
    return {"ok": True}
```

- [ ] **Step 4: Run test, verify pass**

Run: `uv run pytest tests/test_api.py::test_health_returns_ok -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add api.py tests/test_api.py
git commit -m "feat(api): bootstrap FastAPI app with /health"
```

---

## Task 3: Add `X-API-Key` auth dependency (TDD)

**Files:**
- Modify: `api.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_api.py`:

```python
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


def test_invoke_with_valid_key_reaches_handler(client, api_headers):
    # We haven't wired the graph yet, so the endpoint returns a stub.
    # This test only proves auth passes; response shape is tested in Task 4.
    response = client.post("/invoke", json={"query": "hi"}, headers=api_headers)
    assert response.status_code != 401
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_api.py -v -k "invoke"`
Expected: all three FAIL — the first two with 404 (route not defined), the third also 404.

- [ ] **Step 3: Add auth dependency and stub `/invoke`**

Edit `api.py`. After the `app = FastAPI(...)` line, add:

```python
import os
from typing import Annotated

from fastapi import Depends, Header, HTTPException
from pydantic import BaseModel, Field


def require_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    if x_api_key != os.environ.get("API_KEY"):
        raise HTTPException(status_code=401, detail="invalid api key")


class InvokeRequest(BaseModel):
    query: str = Field(min_length=1)


@app.post("/invoke", dependencies=[Depends(require_api_key)])
def invoke(request: InvokeRequest) -> dict:
    # Stub — wired to graph in Task 4.
    return {"answer": "stub", "place_id_map": {}, "recommended_questions": []}
```

Consolidate imports at the top of the file (keep `from __future__ import annotations` first). Final import block should look like:

```python
from __future__ import annotations

import os
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_api.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add api.py tests/test_api.py
git commit -m "feat(api): add X-API-Key auth dependency"
```

---

## Task 4: Wire `/invoke` to the graph (TDD)

**Files:**
- Modify: `api.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_api.py`:

```python
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
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_api.py -v -k "invoke_returns_response_shape or invoke_missing_optional"`
Expected: both FAIL — response body is the stub, not the mocked values (and the second returns stub instead of `{answer: "just an answer", ...}`).

The `_returns_422` tests may already pass (Pydantic + `min_length=1` enforce that), but running them confirms the constraint. Add to the run: `-k "invoke"`.

- [ ] **Step 3: Wire the graph and response model**

Edit `api.py`. Add the graph import and the `InvokeResponse` model. Replace the stub handler with the real one.

Add near the top imports:

```python
from langchain_core.messages import HumanMessage

from agents.places.workflow import graph
```

Add after `InvokeRequest`:

```python
class InvokeResponse(BaseModel):
    answer: str
    place_id_map: dict[str, list[str]] = Field(default_factory=dict)
    recommended_questions: list[str] = Field(default_factory=list)
```

Replace the `invoke` handler:

```python
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
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_api.py -v`
Expected: all `invoke` tests PASS.

- [ ] **Step 5: Add failing test for graph-exception 500 shape**

Append to `tests/test_api.py`:

```python
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
```

Run: `uv run pytest tests/test_api.py::test_invoke_graph_exception_returns_500_with_detail -v`
Expected: FAIL — TestClient raises the RuntimeError, or returns FastAPI's default 500 with `"Internal Server Error"`.

- [ ] **Step 6: Add global exception handler**

Edit `api.py`. Add import (Request):

```python
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
```

Add handler below the `app = FastAPI(...)` line:

```python
@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": str(exc)})
```

(The `logger` symbol is added in Task 5. For Task 4, add it now as well to avoid a forward reference:)

```python
import logging

logger = logging.getLogger(__name__)
```

- [ ] **Step 7: Run tests, verify pass**

Run: `uv run pytest tests/test_api.py -v`
Expected: all tests including the new 500-shape test PASS.

- [ ] **Step 8: Commit**

```bash
git add api.py tests/test_api.py
git commit -m "feat(api): wire /invoke to graph with response model and 500 handler"
```

---

## Task 5: Add `/stream` SSE endpoint (TDD)

**Files:**
- Modify: `api.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_api.py`:

```python
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
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_api.py -v -k "stream"`
Expected: FAIL — `/stream` route not defined (404) or auth passes but 405/404.

- [ ] **Step 3: Implement `/stream`**

Edit `api.py`. Add imports (extend `fastapi.responses` line, and add lc_dumps):

```python
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.load import dumps as lc_dumps
```

(`logging` and `logger` were added in Task 4.)

Add helper below `InvokeResponse`:

```python
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
```

Add the endpoint after `invoke`:

```python
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
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_api.py -v`
Expected: all `stream` tests PASS, plus the previous 8 continue to pass.

- [ ] **Step 5: Commit**

```bash
git add api.py tests/test_api.py
git commit -m "feat(api): add /stream SSE endpoint"
```

---

## Task 6: CORS + manual smoke test

**Files:**
- Modify: `api.py`

- [ ] **Step 1: Add CORS middleware**

Edit `api.py`. Add import:

```python
from fastapi.middleware.cors import CORSMiddleware
```

Add immediately after `app = FastAPI(...)`:

```python
# TODO: restrict for prod. Local dev opens everything.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

- [ ] **Step 2: Add CORS preflight test**

Append to `tests/test_api.py`:

```python
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
```

- [ ] **Step 3: Run full suite**

Run: `uv run pytest tests/test_api.py -v`
Expected: all tests PASS.

- [ ] **Step 4: Start the server manually and smoke test**

In one terminal:

```bash
uv run uvicorn api:app --reload --port 8000
```

Expected: startup log ends with `Uvicorn running on http://127.0.0.1:8000`.

In another terminal:

```bash
curl http://localhost:8000/health
```

Expected: `{"ok":true}`.

```bash
API_KEY=$(grep '^API_KEY=' .env | cut -d= -f2)
curl -sS -X POST http://localhost:8000/invoke \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"베들레헴은 어디야?"}'
```

Expected: JSON with `answer`, `place_id_map`, `recommended_questions`. `place_id_map` should contain at least one Bethlehem entry.

```bash
curl -N -X POST http://localhost:8000/stream \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"베들레헴은 어디야?"}'
```

Expected: multiple `event: node` lines followed by one `event: done`.

Kill the server with Ctrl+C.

- [ ] **Step 5: Commit**

```bash
git add api.py tests/test_api.py
git commit -m "feat(api): add CORS middleware and preflight test"
```

---

## Post-plan verification

After Task 6:

- [ ] `uv run pytest tests/ -v` — full project suite still passes.
- [ ] `git log --oneline main..HEAD` — six new commits (Tasks 1-6) plus the spec commit from earlier.
- [ ] `api.py` is under ~200 LoC. If not, revisit the "single-file" decision before opening the PR.

Not covered by this plan (from spec's Out of Scope): Dockerfile, multi-turn state, metrics, rate limiting, restricted CORS origins, projection helper for streaming payloads, second agent exposure.
