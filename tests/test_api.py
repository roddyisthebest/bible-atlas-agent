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
