"""Name-based retrieval regression test (pytest).

tests/fixtures/sample_places.json 의 각 레코드에 대해:

- parent: `search_ancient_places(names=[name])` 로 조회 시 자기 자신이 결과에 있어야 함
- child : `search_modern_places(names=[name])` 로 조회 시 자기 자신이 결과에 있어야 함
- English name (`name`) 과 Korean name (`koreanName`) 각각 검증

fixture 는 리포에 커밋돼 있어 GitHub CI 에서도 원본 data/ai-places-data/
없이 실행 가능. 단, Postgres DB 에는 fixture 의 id 들이 존재해야 함.

실행:
    uv run pytest tests/test_place_retrieval.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.places.place_tools import (
    search_ancient_places,
    search_modern_places,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "sample_places.json"
TOP_K = 5


# ---------------------------------------------------------------------------
# Fixture load
# ---------------------------------------------------------------------------
def _load_records() -> list[dict]:
    if not FIXTURE_PATH.exists():
        pytest.skip(f"fixture not found: {FIXTURE_PATH}")
    with FIXTURE_PATH.open(encoding="utf-8") as f:
        return json.load(f)["data"]


_records = _load_records()
_parents = [r for r in _records if r.get("stereo") == "parent" and r.get("id")]
_children = [r for r in _records if r.get("stereo") == "child" and r.get("id")]


def _record_id(record: dict) -> str:
    return f"{record.get('id')}-{record.get('koreanName') or record.get('name')}"


# ---------------------------------------------------------------------------
# Parent lookup
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("record", _parents, ids=_record_id)
def test_parent_lookup_by_english_name(record):
    name = record.get("name")
    if not name:
        pytest.skip("no english name")

    result = search_ancient_places.invoke({
        "names": [name],
        "top_k_per_name": TOP_K,
    })
    returned_ids = [r["place_id"] for r in result]
    assert record["id"] in returned_ids, (
        f"parent {record['id']} ({name}) not found by english name. "
        f"got ids: {returned_ids}"
    )


@pytest.mark.parametrize("record", _parents, ids=_record_id)
def test_parent_lookup_by_korean_name(record):
    name = record.get("koreanName")
    if not name:
        pytest.skip("no korean name")

    result = search_ancient_places.invoke({
        "names": [name],
        "top_k_per_name": TOP_K,
    })
    returned_ids = [r["place_id"] for r in result]
    assert record["id"] in returned_ids, (
        f"parent {record['id']} ({name}) not found by korean name. "
        f"got ids: {returned_ids}"
    )


# ---------------------------------------------------------------------------
# Child lookup
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("record", _children, ids=_record_id)
def test_child_lookup_by_english_name(record):
    name = record.get("name")
    if not name:
        pytest.skip("no english name")

    result = search_modern_places.invoke({
        "names": [name],
        "top_k_per_name": TOP_K,
    })
    returned_ids = [r["place_id"] for r in result]
    assert record["id"] in returned_ids, (
        f"child {record['id']} ({name}) not found by english name. "
        f"got ids: {returned_ids}"
    )


@pytest.mark.parametrize("record", _children, ids=_record_id)
def test_child_lookup_by_korean_name(record):
    name = record.get("koreanName")
    if not name:
        pytest.skip("no korean name")

    result = search_modern_places.invoke({
        "names": [name],
        "top_k_per_name": TOP_K,
    })
    returned_ids = [r["place_id"] for r in result]
    assert record["id"] in returned_ids, (
        f"child {record['id']} ({name}) not found by korean name. "
        f"got ids: {returned_ids}"
    )
