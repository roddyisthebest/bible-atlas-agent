"""FastAPI service exposing agents.places.workflow.graph.

Endpoints:
    GET  /health   — liveness (no auth)
    POST /invoke   — sync agent call (X-API-Key)
    POST /stream   — SSE stream of graph updates (X-API-Key)
"""
from __future__ import annotations

import os
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(title="bible-atlas-agent")


def require_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    if x_api_key != os.environ.get("API_KEY"):
        raise HTTPException(status_code=401, detail="invalid api key")


class InvokeRequest(BaseModel):
    query: str = Field(min_length=1)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/invoke", dependencies=[Depends(require_api_key)])
def invoke(request: InvokeRequest) -> dict:
    # Stub — wired to graph in Task 4.
    return {"answer": "stub", "place_id_map": {}, "recommended_questions": []}
