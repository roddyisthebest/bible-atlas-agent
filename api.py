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
