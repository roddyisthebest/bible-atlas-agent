# FastAPI Service for Bible Atlas Agent — Design

## Overview

The bible-atlas agent currently runs only inside notebooks or via LangGraph
Platform deployment (`agents/places/workflow.py:graph`). We want a lightweight
HTTP service that exposes the same graph so a frontend (or curl) can call it
directly during local development.

- **Scope**: local dev only. Single-shot request/response and streaming
  variants of the same underlying graph.
- **Non-goals**: docker/deploy scripts, multi-turn/thread state, rate
  limiting, prod-grade auth, request logging/metrics, restricted CORS,
  streaming payload projection helper.

## Architecture

- Single file `api.py` at the project root (< ~150 LoC target).
- FastAPI app object at `api:app`.
- Imports and reuses `agents.places.workflow.graph` unmodified.
- Stateless: each request builds `initial_state = {"query": q, "messages": [HumanMessage(q)]}`
  and runs the graph. No session/thread management.

## Endpoints

| Method | Path      | Auth | Body                | Response                                                     |
|--------|-----------|------|---------------------|--------------------------------------------------------------|
| GET    | `/health` | ❌   | —                   | `{"ok": true}`                                               |
| POST   | `/invoke` | ✅   | `{"query": str}`    | `{"answer": str, "place_id_map": {...}, "recommended_questions": [...]}` |
| POST   | `/stream` | ✅   | `{"query": str}`    | `text/event-stream` (see Streaming section)                  |

## Auth

Single shared secret via `X-API-Key` header, compared to `os.environ["API_KEY"]`.

```python
def require_api_key(x_api_key: Annotated[str | None, Header()] = None):
    if x_api_key != os.environ["API_KEY"]:
        raise HTTPException(401, "invalid api key")
```

Applied via `Depends(require_api_key)` on `/invoke` and `/stream`. `/health`
stays open.

## Request / Response Schemas

Pydantic models in `api.py`:

- `InvokeRequest`: `{"query": str}` — `query` has `min_length=1`.
- `InvokeResponse`: `{"answer": str, "place_id_map": dict[str, list[str]],
  "recommended_questions": list[str]}` — matches the terminal state fields
  returned by the `format` / `non_bible_reject` / `bible_general_agent` nodes.
  `recommended_questions` defaults to `[]` when the graph did not populate it.

## Streaming (`/stream`)

Uses `graph.stream(initial_state, stream_mode="updates")`, which yields
`dict[node_name, update_delta]` after each node.

Three SSE event types:

```
event: node
data: {"node": "router", "update": null}

event: node
data: {"node": "place_agent", "update": {"messages": [ ... ]}}

event: done
data: {"answer": "...", "place_id_map": {...}, "recommended_questions": []}

event: error
data: {"detail": "..."}
```

- `node`: emitted for every step in the graph stream. `update` is the raw
  LangGraph delta.
- `done`: emitted once at the end, carries the same shape as `/invoke`'s
  response.
- `error`: emitted if the graph raises. Generator then stops.

**Serialization**: LangGraph updates contain `AIMessage`, `ToolMessage` and
similar LangChain objects that plain `json.dumps` cannot handle. Use
`langchain_core.load.dumps(obj)` which emits JSON with LangChain type
markers. Clients that don't know LangChain can still read `content` /
`tool_calls` fields — the markers only add extra keys.

**Response headers**:
- `Content-Type: text/event-stream`
- `Cache-Control: no-cache`
- `X-Accel-Buffering: no` (defensive; only meaningful behind nginx)

**Client disconnect**: FastAPI stops iterating the generator, but the
underlying `graph.stream` call may complete in the background. Acceptable
for dev use; a proper cancellation path is out of scope.

## CORS

`CORSMiddleware` with `allow_origins=["*"]`, `allow_methods=["*"]`,
`allow_headers=["*"]`. A `# TODO: restrict for prod` comment marks the
line so it is impossible to miss when hardening.

## Error Handling

- **401** — invalid or missing `X-API-Key` (from `require_api_key`).
- **422** — request body validation (FastAPI default via Pydantic).
- **500** — graph raises unexpected exception. Response body is
  `{"detail": str(exc)}`; full traceback goes to server logs via `logging`.

No custom exception hierarchy for MVP.

## Config

- `.env` at project root is loaded by both `agents.places.workflow` and
  `agents.places.place_tools` at import time via `load_dotenv()`. `api.py`
  also calls `load_dotenv()` at module top so `API_KEY` is available
  before FastAPI initialization regardless of import order.
- New env var: `API_KEY` (single shared secret). Generate with
  `openssl rand -hex 32`.

## Testing

`tests/test_api.py`, FastAPI `TestClient`, `monkeypatch` on `api.graph` so
no real LLM or DB calls happen:

- `GET /health` → 200, `{"ok": true}`
- `POST /invoke` **no** `X-API-Key` → 401
- `POST /invoke` **wrong** key → 401
- `POST /invoke` valid key + mocked `graph.invoke` → response shape matches
  `InvokeResponse`
- `POST /stream` valid key + mocked `graph.stream` (yields fake updates) →
  raw SSE body parses into the expected sequence of `node` events followed
  by a single `done`
- `POST /invoke` with empty body → 422

Mocking pattern:

```python
monkeypatch.setattr(
    "api.graph",
    SimpleNamespace(
        invoke=lambda state: {
            "answer": "...",
            "place_id_map": {},
            "recommended_questions": [],
        },
        stream=lambda state, stream_mode: iter([
            {"router": None},
            {"place_agent": {"messages": []}},
            {"format": {"answer": "...", "place_id_map": {}}},
        ]),
    ),
)
```

## Dependencies (pyproject.toml additions)

```toml
"fastapi>=0.115.0",
"uvicorn[standard]>=0.32.0",
```

`pytest` and `httpx` (transitive via fastapi's TestClient extra) already
present or auto-installed by uv.

## Run

```bash
uv run uvicorn api:app --reload --port 8000
```

Smoke:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/invoke \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"베들레헴은 어디야?"}'
```

## Branch

`feat/fastapi-service`, branched from `main`. Spec + implementation live on
this branch and merge back via PR.

## Out of Scope (explicit)

- Dockerfile / deployment scripts
- Multi-turn / thread_id / checkpointer
- Request logging beyond default uvicorn access log
- Metrics, tracing, OpenTelemetry
- Rate limiting
- Restricted CORS origins
- Streaming payload projection helper (raw `lc_dumps` for now)
- Any second agent — only `agents.places.workflow.graph` is exposed
