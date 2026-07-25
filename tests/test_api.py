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
