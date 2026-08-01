import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field

load_dotenv()

POSTGRES_URL = os.environ["POSTGRES_URL"]

# ---------------------------------------------------------------------------
# Postgres connection pool
# ---------------------------------------------------------------------------
# 프로세스당 pool 1개. 각 tool 호출은 pool.connection()으로 짧게 borrow.
# min_size/max_size는 소규모 트래픽 기준. 필요 시 상향.
#
# managed DB (Railway/Neon/Supabase 등) 는 유지보수·idle timeout·admin
# terminate 로 pool 안 idle 커넥션을 몰래 죽인다. 다음 checkout 이 죽은
# 커넥션을 잡으면 "terminating connection due to administration command"
# (SQLSTATE 57P01) 가 사용자에게 그대로 노출됨.
#
# 방어:
#   - check=check_connection: 넘겨주기 직전에 짧은 ping 으로 살아있는지
#     검증, 죽었으면 pool 이 자동으로 교체.
#   - max_idle=300: 5분 이상 놀린 커넥션은 서버 idle timeout 걸리기
#     전에 미리 축출.
#   - max_lifetime=1800: 30분마다 강제 재생성, 장수 커넥션이 서버 재시작
#     시점을 넘어가는 상황 방지.
_pool = ConnectionPool(
    conninfo=POSTGRES_URL,
    min_size=1,
    max_size=5,
    max_idle=300,
    max_lifetime=1800,
    check=ConnectionPool.check_connection,
    open=True,   # 모듈 로드 시 즉시 열기
)


keyword_llm = ChatOpenAI(model="gpt-5.6-luna",temperature=0)
journey_llm = ChatOpenAI(model="gpt-4o-mini")

keyword_search_prompt = PromptTemplate.from_template(
    """
사용자의 질문에서 정답일 가능성이 높은 성경 장소 엔티티를
최대 3개까지 반환하세요.

도시, 지역뿐 아니라 신전, 건물, 산, 강, 광야, 우물 등도 포함합니다.
질문의 직접적인 정답을 먼저 반환하고, 필요하면 관련 상위 지역을 추가하세요.
장소를 특정할 수 없으면 빈 목록을 반환하세요.

**출력 규칙 (매우 중요):**
- 장소 이름은 반드시 **영어 성경 표준 표기**로 출력합니다.
  한국어 이름(도단·베들레헴 등)이 아니라 영어(Dothan·Bethlehem 등)로 반환하세요.
- 이유: DB에서 이름 매칭이 안정적으로 되기 위함입니다. 한국어는 표기 변형이
  많아(안티오크/안티옥/안디옥 등) 매칭 실패가 잦습니다.
- 개역개정 성경의 영어 병기 이름 또는 널리 알려진 영어 성경(NIV, ESV 등)의
  표기를 사용하세요.

예시:
Q: 요셉이 형들에게 팔린 곳은?
A: ["Dothan"]

Q: 예수님이 사마리아 여인과 대화를 나눈 우물은 어디인가요?
A: ["Jacob's Well", "Sychar"]

Q: 여로보암이 금송아지를 세운 최북단 성읍은?
A: ["Dan"]

Q: 천국은 어디에 있나요?
A: []

질문: {query}
"""
)


class KeywordResult(BaseModel):
    keywords: list[str] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "질문의 정답 또는 근거와 관련된 성경 장소 엔티티명. "
            "**반드시 영어 성경 표준 표기**로 반환한다 (예: 'Bethlehem', 'Antioch', 'Cyprus'). "
            "도시, 성읍, 지역, 산, 강, 광야, 신전, 건물, 우물 등을 포함한다. "
            "직접적인 정답을 먼저 반환하고, 필요하면 상위 지역명을 추가한다. "
            "특정할 수 없으면 빈 배열을 반환한다."
        ),
    )


keyword_search_chain = keyword_search_prompt | keyword_llm.with_structured_output(
    KeywordResult
)


journey_route_search_prompt = PromptTemplate.from_template(
    """
사용자의 질문에서 성경 속 인물이나 집단의 이동 여정을 파악하고,
관련 장소명을 실제 이동 순서대로 최대 10개까지 반환하세요.

규칙:
- 도시, 지역, 산, 강, 광야, 항구, 건물 등 이동 경로와 관련된 장소를 포함하세요.
- 반드시 출발지부터 도착지까지 순서대로 정렬하세요.
- 질문에서 요구한 여정 구간에 해당하는 장소만 반환하세요.
- 실제로 같은 장소를 다시 방문했다면 중복을 유지하세요.
- 이동 순서를 신뢰할 수 없거나 여정 질문이 아니면 빈 목록을 반환하세요.
- 설명 없이 장소명 배열만 반환하세요.

**출력 규칙 (매우 중요):**
- 장소 이름은 반드시 **영어 성경 표준 표기**로만 출력합니다.
  한국어(안디옥·구브로 등)나 음차(안티오크·키프로스 등)는 절대 사용 금지.
  반드시 영어(Antioch·Cyprus·Salamis·Paphos·Perga·Iconium·Lystra·Derbe 등)로 반환.
- 이유: DB에서 이름 매칭이 안정적으로 되기 위함입니다. 한국어 표기는 변형이
  많아(안디옥/안티옥/안티오크 등) 매칭이 자주 실패합니다.
- 개역개정 성경의 영어 병기 이름 또는 널리 알려진 영어 성경(NIV, ESV 등)의
  표기를 사용하세요.

**추상적·개념적 지명 금지 (매우 중요):**
- 성경에 실제로 등장하는 **구체적이고 고유한 지명**만 사용하세요.
- 아래처럼 개념·상징·목적지를 지칭하는 추상 표현은 **절대 포함하지 마세요**:
  - "Promised Land" / "Land of Promise" → "Canaan"으로 대체
  - "the wilderness" / "the desert" (일반) → 구체적 광야명 사용
    (예: "Wilderness of Shur", "Wilderness of Sin", "Wilderness of Paran")
  - "Heaven", "Hell", "Paradise" 같은 초월적 장소
  - "홈타운", "고향", "이스라엘 땅" 같은 지시어
- 판단 기준: 그 이름을 성경 원문에서 고유명사처럼 찾을 수 있어야 함.
  못 찾으면 빼거나 구체적 이름으로 교체.

예시:
Q: 바울이 1차 전도여행 중 구브로에서 방문한 도시 순서는?
A: ["Salamis", "Paphos"]

Q: 바울의 2차 전도여행 여정 순서는?
A: ["Antioch", "Syria", "Cilicia", "Derbe", "Lystra", "Iconium", "Phrygia", "Galatia", "Troas", "Philippi"]

Q: 출애굽 여정을 알려줘
A: ["Egypt", "Red Sea", "Wilderness of Shur", "Marah", "Elim", "Wilderness of Sin", "Rephidim", "Mount Sinai", "Kadesh-barnea", "Canaan"]
  # "Promised Land"는 사용 금지, "Canaan"으로 대체함.

Q: 요셉이 형들에게 팔린 곳은 어디인가요?
A: []

Q: 천국은 어디에 있나요?
A: []

질문: {query}
답변:
"""
)


class JourneyRouteResult(BaseModel):
    places: list[str] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "성경 인물이나 집단이 이동한 장소를 출발지부터 도착지까지 "
            "실제 이동 순서대로 나열한 목록. "
            "**반드시 영어 성경 표준 표기**로 반환한다 (예: 'Antioch', 'Cyprus', 'Salamis'). "
            "한국어·음차 표기는 금지. 여정이 아니면 빈 배열."
        ),
    )


journey_route_search_chain = (
    journey_route_search_prompt | journey_llm.with_structured_output(JourneyRouteResult)
)


journey_description_search_prompt = PromptTemplate.from_template(
    """
사용자의 질문에서 언급되거나 암시된 성경 여정을 파악하고,
그 여정에 대한 간결한 서술적 설명을 작성하세요.

규칙:
- 여정의 주요 인물(또는 집단), 시대적·역사적 배경, 목적, 신학적·성경적 의미를
  중심으로 3~5문장 이내로 요약하세요.
- 구체적인 장소들의 이동 순서는 별도 도구(journey_route_search)에서 다루므로
  나열 형식이 아닌 서술 형식으로 작성하세요.
- 여정이 아니거나 여정을 특정할 수 없으면 빈 문자열을 반환하세요.

예시:
질문: 바울의 1차 전도여행은 어떤 여행이었어?
답변: 바울과 바나바가 안디옥 교회의 파송을 받아 떠난 최초의 이방 선교 여행이다. 구브로 섬을 시작으로 소아시아 남부의 여러 도시에서 복음을 전했으며, 이방인에게도 구원의 문이 열려 있음을 보여준 결정적 사건으로 평가된다.

질문: 요셉이 형들에게 팔린 곳은 어디인가요?
답변:

질문: 천국은 어디에 있나요?
답변:

질문: {query}
답변:
"""
)


class JourneyDescriptionResult(BaseModel):
    description: str = Field(
        default="",
        description=(
            "성경 여정의 배경·목적·의미를 담은 짧은 서술 텍스트. "
            "여정이 아니거나 특정할 수 없으면 빈 문자열."
        ),
    )


journey_description_search_chain = (
    journey_description_search_prompt
    | journey_llm.with_structured_output(JourneyDescriptionResult)
)


class PlaceSearchResult(TypedDict):
    place_id: str
    name_ko: str | None
    name_en: str | None
    types: list[str] | None
    bible_references: list[str] | None
    identification_names: list[str] | None
    text: str | None


class ModernPlaceRecord(TypedDict):
    place_id: str
    name_ko: str | None
    name_en: str | None
    types: list[str] | None
    parent_ids: list[str] | None
    text: str | None


class AncientWithModernResult(TypedDict):
    """search_ancient_with_modern 반환 타입. parent + join 된 modern candidates."""
    place_id: str
    name_ko: str | None
    name_en: str | None
    types: list[str] | None
    bible_references: list[str] | None
    identification_names: list[str] | None
    text: str | None
    modern_places: list[ModernPlaceRecord]


class ModernWithAncientResult(TypedDict):
    """search_modern_with_ancient 반환 타입. child + join 된 ancient parents."""
    place_id: str
    name_ko: str | None
    name_en: str | None
    types: list[str] | None
    text: str | None
    ancient_places: list[PlaceSearchResult]


# ---------------------------------------------------------------------------
# Postgres query for name-based lookup
# ---------------------------------------------------------------------------
# 매칭 규칙: korean_name / name에 keyword가 포함되면 매치 (ILIKE '%kw%').
#   - "베들레헴" → "베들레헴", "베들레헴 1", "베들레헴 1 (유다)" 등 다 잡힘
#   - GIN trigram 인덱스(places_*_trgm_idx)가 contains 검색을 가속함
# 정렬: 완전 일치 → 시작 일치 → 이름 길이 짧은 순.
_LOOKUP_SQL = """
SELECT
    id,
    korean_name,
    name,
    types,
    verses,
    korean_description,
    description,
    identification_names,
    parent_ids
FROM places
WHERE stereo = %(stereo)s
  AND (
        korean_name ILIKE '%%' || %(kw)s || '%%'
     OR name ILIKE '%%' || %(kw)s || '%%'
  )
ORDER BY
    (korean_name = %(kw)s OR LOWER(name) = LOWER(%(kw)s)) DESC,
    (korean_name ILIKE %(kw)s || '%%' OR name ILIKE %(kw)s || '%%') DESC,
    LENGTH(COALESCE(korean_name, name))
LIMIT %(limit)s
"""


def _lookup_by_name(name: str, stereo: str, limit: int) -> list[tuple]:
    """이름으로 places 테이블 조회."""
    with _pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                _LOOKUP_SQL,
                {"stereo": stereo, "kw": name, "limit": limit},
            )
            return cur.fetchall()


# ---------------------------------------------------------------------------
# JOIN queries — 한 왕복으로 parent+children (또는 child+parents) 함께 조회
# ---------------------------------------------------------------------------
_ANCIENT_WITH_MODERN_SQL = """
SELECT
    p.id,
    p.korean_name,
    p.name,
    p.types,
    p.verses,
    p.korean_description,
    p.description,
    p.identification_names,
    COALESCE(
        (
            SELECT json_agg(
                json_build_object(
                    'place_id', c.id,
                    'name_ko', c.korean_name,
                    'name_en', c.name,
                    'types', c.types,
                    'parent_ids', c.parent_ids,
                    'text', COALESCE(c.korean_description, c.description)
                )
            )
            FROM places c
            WHERE c.stereo = 'child'
              AND c.id = ANY(p.identification_ids)
        ),
        '[]'::json
    ) AS modern_places
FROM places p
WHERE p.stereo = 'parent'
  AND (
        p.korean_name ILIKE '%%' || %(kw)s || '%%'
     OR p.name ILIKE '%%' || %(kw)s || '%%'
  )
ORDER BY
    (p.korean_name = %(kw)s OR LOWER(p.name) = LOWER(%(kw)s)) DESC,
    (p.korean_name ILIKE %(kw)s || '%%' OR p.name ILIKE %(kw)s || '%%') DESC,
    LENGTH(COALESCE(p.korean_name, p.name))
LIMIT %(limit)s
"""


_MODERN_WITH_ANCIENT_SQL = """
SELECT
    c.id,
    c.korean_name,
    c.name,
    c.types,
    c.korean_description,
    c.description,
    c.parent_ids,
    COALESCE(
        (
            SELECT json_agg(
                json_build_object(
                    'place_id', p.id,
                    'name_ko', p.korean_name,
                    'name_en', p.name,
                    'types', p.types,
                    'bible_references', p.verses,
                    'identification_names', p.identification_names,
                    'text', COALESCE(p.korean_description, p.description)
                )
            )
            FROM places p
            WHERE p.stereo = 'parent'
              AND p.id = ANY(c.parent_ids)
        ),
        '[]'::json
    ) AS ancient_places
FROM places c
WHERE c.stereo = 'child'
  AND (
        c.korean_name ILIKE '%%' || %(kw)s || '%%'
     OR c.name ILIKE '%%' || %(kw)s || '%%'
  )
ORDER BY
    (c.korean_name = %(kw)s OR LOWER(c.name) = LOWER(%(kw)s)) DESC,
    (c.korean_name ILIKE %(kw)s || '%%' OR c.name ILIKE %(kw)s || '%%') DESC,
    LENGTH(COALESCE(c.korean_name, c.name))
LIMIT %(limit)s
"""


def _lookup_ancient_with_modern(name: str, limit: int) -> list[tuple]:
    with _pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_ANCIENT_WITH_MODERN_SQL, {"kw": name, "limit": limit})
            return cur.fetchall()


def _lookup_modern_with_ancient(name: str, limit: int) -> list[tuple]:
    with _pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_MODERN_WITH_ANCIENT_SQL, {"kw": name, "limit": limit})
            return cur.fetchall()


# ---------------------------------------------------------------------------
# Tools — 기본 이름 검색 (고대 / 현대)
# ---------------------------------------------------------------------------
@tool
def search_ancient_places(
    names: list[str],
    top_k_per_name: int = 3,
) -> list[PlaceSearchResult]:
    """고대 성경 장소 이름 목록으로 지명 데이터를 조회합니다.

    각 이름에 대해 Postgres `places` 테이블(stereo='parent')에서
    korean_name/name 의 contains 매칭(ILIKE %name%)으로 조회합니다.
    disambiguation 접미(예: "Bethlehem 1", "베들레헴 1 (유다)")도 자동 매칭.

    **중요: names 는 반드시 영어 성경 표준 표기로만 넘겨야 합니다.**
    한국어·음차(안디옥/안티오크/키프로스 등)는 표기 변형이 많아 실패율이 높음.
    영어(Antioch, Cyprus, Bethlehem, Salamis 등)를 사용하세요.

    Args:
        names: 조회할 고대 성경 지명 이름 목록. **반드시 영어**. 최대 20개까지 처리.
        top_k_per_name: 이름별 반환 후보 수. 기본값 3, 1~5 사이 제한.

    Returns:
        고대 지명의 place_id, 한/영 이름, 지명 유형, 관련 성경 구절,
        현대 위치 후보 이름(identification_names), 본문 설명.
        place_id 기준 중복 제거.
    """
    normalized = [n.strip() for n in names if n and n.strip()]
    if not normalized:
        return []
    limited = normalized[:20]
    top_k = max(1, min(top_k_per_name, 5))

    seen: set[str] = set()
    results: list[PlaceSearchResult] = []
    for name in limited:
        for row in _lookup_by_name(name, "parent", top_k):
            (
                place_id,
                korean_name,
                name_en,
                types,
                verses,
                korean_description,
                description,
                identification_names,
                _parent_ids,
            ) = row
            if place_id in seen:
                continue
            seen.add(place_id)
            results.append(
                PlaceSearchResult(
                    place_id=place_id,
                    name_ko=korean_name,
                    name_en=name_en,
                    types=list(types) if types else None,
                    bible_references=list(verses) if verses else None,
                    identification_names=list(identification_names) if identification_names else None,
                    text=korean_description or description,
                )
            )
    return results


@tool
def search_modern_places(
    names: list[str],
    top_k_per_name: int = 3,
) -> list[ModernPlaceRecord]:
    """현대 지명 이름 목록으로 현대 지명 레코드를 조회합니다.

    각 이름에 대해 Postgres `places` 테이블(stereo='child')에서
    korean_name/name 의 contains 매칭(ILIKE %name%)으로 조회합니다.

    Args:
        names: 조회할 현대 지명 이름 목록. 최대 20개까지 처리.
        top_k_per_name: 이름별 반환 후보 수. 기본값 3, 1~5 사이 제한.

    Returns:
        현대 지명의 place_id, 한/영 이름, 지명 유형,
        연관된 고대 성경 장소 ID 목록(parent_ids), 본문 설명.
        place_id 기준 중복 제거.
    """
    normalized = [n.strip() for n in names if n and n.strip()]
    if not normalized:
        return []
    limited = normalized[:20]
    top_k = max(1, min(top_k_per_name, 5))

    seen: set[str] = set()
    records: list[ModernPlaceRecord] = []
    for name in limited:
        for row in _lookup_by_name(name, "child", top_k):
            (
                place_id,
                korean_name,
                name_en,
                types,
                _verses,
                korean_description,
                description,
                _identification_names,
                parent_ids,
            ) = row
            if place_id in seen:
                continue
            seen.add(place_id)
            records.append(
                ModernPlaceRecord(
                    place_id=place_id,
                    name_ko=korean_name,
                    name_en=name_en,
                    types=list(types) if types else None,
                    parent_ids=list(parent_ids) if parent_ids else None,
                    text=korean_description or description,
                )
            )
    return records


# ---------------------------------------------------------------------------
# Tools — JOIN 조회 (한 왕복으로 양방향 record 함께)
# ---------------------------------------------------------------------------
@tool
def search_ancient_with_modern(
    names: list[str],
    top_k_per_name: int = 3,
) -> list[AncientWithModernResult]:
    """고대 지명 이름으로 조회하되, 각 고대 지명과 연결된 현대 위치 후보 레코드를
    한 번의 쿼리로 함께 반환합니다.

    "이 고대 지명은 지금 어디에 있어?" 처럼 고대 지명 정보와 현대 위치 후보들을
    동시에 필요할 때 이 도구 하나로 두 번의 조회를 대체합니다.

    Args:
        names: 조회할 고대 성경 지명 이름 목록. **반드시 영어**. 최대 20개.
        top_k_per_name: 이름별 반환 parent 후보 수. 기본값 3, 1~5 제한.

    Returns:
        각 parent record 필드 + `modern_places` (연결된 child record 목록).
        place_id 기준 중복 제거.
    """
    normalized = [n.strip() for n in names if n and n.strip()]
    if not normalized:
        return []
    limited = normalized[:20]
    top_k = max(1, min(top_k_per_name, 5))

    seen: set[str] = set()
    results: list[AncientWithModernResult] = []
    for name in limited:
        for row in _lookup_ancient_with_modern(name, top_k):
            (
                place_id,
                korean_name,
                name_en,
                types,
                verses,
                korean_description,
                description,
                identification_names,
                modern_places_json,
            ) = row
            if place_id in seen:
                continue
            seen.add(place_id)
            modern_places = [
                ModernPlaceRecord(**m) for m in (modern_places_json or [])
            ]
            results.append(
                AncientWithModernResult(
                    place_id=place_id,
                    name_ko=korean_name,
                    name_en=name_en,
                    types=list(types) if types else None,
                    bible_references=list(verses) if verses else None,
                    identification_names=list(identification_names) if identification_names else None,
                    text=korean_description or description,
                    modern_places=modern_places,
                )
            )
    return results


@tool
def search_modern_with_ancient(
    names: list[str],
    top_k_per_name: int = 3,
) -> list[ModernWithAncientResult]:
    """현대 지명 이름으로 조회하되, 각 현대 지명과 연결된 고대 성경 지명 후보 레코드를
    한 번의 쿼리로 함께 반환합니다.

    "이 현대 지명이 성경 어디에 해당해?" 처럼 현대 지명 정보와 고대 후보들을
    동시에 필요할 때 이 도구 하나로 두 번의 조회를 대체합니다.

    Args:
        names: 조회할 현대 지명 이름 목록. 최대 20개.
        top_k_per_name: 이름별 반환 child 후보 수. 기본값 3, 1~5 제한.

    Returns:
        각 child record 필드 + `ancient_places` (연결된 parent record 목록).
        place_id 기준 중복 제거.
    """
    normalized = [n.strip() for n in names if n and n.strip()]
    if not normalized:
        return []
    limited = normalized[:20]
    top_k = max(1, min(top_k_per_name, 5))

    seen: set[str] = set()
    results: list[ModernWithAncientResult] = []
    for name in limited:
        for row in _lookup_modern_with_ancient(name, top_k):
            (
                place_id,
                korean_name,
                name_en,
                types,
                korean_description,
                description,
                _parent_ids,
                ancient_places_json,
            ) = row
            if place_id in seen:
                continue
            seen.add(place_id)
            ancient_places = [
                PlaceSearchResult(**p) for p in (ancient_places_json or [])
            ]
            results.append(
                ModernWithAncientResult(
                    place_id=place_id,
                    name_ko=korean_name,
                    name_en=name_en,
                    types=list(types) if types else None,
                    text=korean_description or description,
                    ancient_places=ancient_places,
                )
            )
    return results


@tool
def journey_route_search(query: str) -> list[str]:
    """성경 여정의 장소 순서(route)를 추출합니다.

    질문에서 언급되거나 암시된 성경 인물·집단의 이동 여정을 파악하고,
    관련 장소들을 출발지부터 도착지까지 이동 순서대로 최대 10개까지
    이름 문자열 리스트로 반환합니다. 여정이 아니거나 특정할 수 없으면
    빈 리스트를 반환합니다.
    """
    result = journey_route_search_chain.invoke({"query": query})
    return result.places


@tool
def journey_description_search(query: str) -> str:
    """성경 여정에 대한 서술적 설명 텍스트를 생성합니다.

    질문에서 언급되거나 암시된 성경 여정의 주요 인물, 시대·역사적 배경,
    목적, 신학적·성경적 의미를 3~5문장의 자연어로 요약해 반환합니다.
    이동 장소의 순서는 다루지 않습니다. 여정이 아니거나 특정할 수 없으면
    빈 문자열을 반환합니다.
    """
    result = journey_description_search_chain.invoke({"query": query})
    return result.description


@tool
def ancient_keyword_search(query: str) -> list[str]:
    """질문에서 정답일 가능성이 높은 고대 성경 장소 이름 후보를 추출합니다.

    사건·인물·지리적 단서를 분석해 관련 장소 엔티티명을 최대 3개까지
    문자열 리스트로 반환합니다. 도시·성읍·지역뿐 아니라 산·강·광야·
    신전·건물·우물 등도 포함됩니다. 장소를 특정할 수 없으면 빈 리스트를
    반환합니다.
    """
    result = keyword_search_chain.invoke({"query": query})
    return result.keywords


__all__ = [
    "PlaceSearchResult",
    "ModernPlaceRecord",
    "AncientWithModernResult",
    "ModernWithAncientResult",
    "KeywordResult",
    "JourneyRouteResult",
    "JourneyDescriptionResult",
    "search_ancient_places",
    "search_modern_places",
    "search_ancient_with_modern",
    "search_modern_with_ancient",
    "journey_route_search",
    "journey_description_search",
    "ancient_keyword_search",
]
