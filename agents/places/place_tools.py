import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pinecone import Pinecone
from pydantic import BaseModel, Field

load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]
PLACES_INDEX_NAME = os.environ["PINECONE_PLACES_INDEX_NAME"]

openai_client = OpenAI()
pc = Pinecone(api_key=PINECONE_API_KEY)
places_index = pc.Index(PLACES_INDEX_NAME)

keyword_llm = ChatOpenAI(model="gpt-5.6-luna")
journey_llm = ChatOpenAI(model="gpt-4o-mini")

keyword_search_prompt = PromptTemplate.from_template(
    """
사용자의 질문에서 정답일 가능성이 높은 성경 장소 엔티티를
최대 3개까지 반환하세요.

도시, 지역뿐 아니라 신전, 건물, 산, 강, 광야, 우물 등도 포함합니다.
질문의 직접적인 정답을 먼저 반환하고, 필요하면 관련 상위 지역을 추가하세요.
장소를 특정할 수 없으면 빈 목록을 반환하세요.

예시:
요셉이 형들에게 팔린 곳은? → 도단
예수님이 사마리아 여인과 대화를 나눈 우물은 어디인가요? → 야곱의 우물, 수가
여로보암이 금송아지를 세운 최북단 성읍은? → 단
천국은 어디에 있나요? → 장소 없음

질문: {query}
"""
)


class KeywordResult(BaseModel):
    keywords: list[str] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "질문의 정답 또는 근거와 관련된 성경 장소 엔티티명. "
            "도시, 성읍, 지역, 산, 강, 광야, 신전, 건물, 우물 등을 포함한다. "
            "직접적인 정답을 먼저 반환하고, 필요하면 상위 지역명을 추가한다. "
            "특정할 수 없으면 빈 배열을 반환한다."
        ),
    )


keyword_search_chain = keyword_search_prompt | keyword_llm.with_structured_output(
    KeywordResult
)


journey_search_prompt = PromptTemplate.from_template(
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

예시:
질문: 바울이 1차 전도여행 중 구브로에서 방문한 도시 순서는?
답변: ["살라미", "바보"]

질문: 요셉이 형들에게 팔린 곳은 어디인가요?
답변: []

질문: 천국은 어디에 있나요?
답변: []

질문: {query}
답변:
"""
)


class JourneyResult(BaseModel):
    places: list[str] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "성경 인물이나 집단이 이동한 장소를 출발지부터 도착지까지 "
            "실제 이동 순서대로 나열한 목록. 여정이 아니면 빈 배열."
        ),
    )


journey_search_chain = journey_search_prompt | journey_llm.with_structured_output(
    JourneyResult
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


@tool
def search_ancient_places(
    keywords: list[str],
    top_k_per_keyword: int = 1,
) -> list[PlaceSearchResult]:
    """고대 성경 장소 이름 키워드 목록으로 지명 데이터를 검색합니다.

    각 키워드마다 벡터 DB(parent 네임스페이스)에서 관련 지명 레코드를
    찾아 반환합니다. 키워드는 다음 세 방식 중 하나로 얻을 수 있습니다.

    1) 질문에 명확한 고대 지명이 이미 들어 있으면 그 지명을 그대로 넘겨
       바로 호출합니다. 예: "베들레헴은 어떤 곳이야?" → keywords=["베들레헴"]
    2) 질문이 사건·인물·지리 단서만 주고 지명을 직접 언급하지 않는다면
       먼저 ancient_keyword_search로 후보 장소명을 추출한 뒤 그 결과를
       그대로 keywords에 넘깁니다.
    3) 질문이 인물·집단의 이동 경로나 여정 순서를 묻는다면 먼저
       journey_search로 여정상의 장소 목록을 얻은 뒤 그 리스트를
       그대로 keywords에 넘겨 각 지점의 지명 정보를 조회합니다.

    다음과 같은 상황에 사용합니다.
    - 질문에 나오는 고대 지명(예: "시글락", "아골 골짜기")의 상세 정보 조회
    - ancient_keyword_search로 추출한 후보 장소명들을 실제 레코드로 확장
    - journey_search가 반환한 여정 장소들의 상세 지명 데이터 조회
    - 여러 고대 지명을 한 번에 묶어 조회

    다음 상황에는 사용하지 마세요.
    - 여정 자체의 이동 순서 확보: journey_search 사용
    - 고대 지명의 현대 위치 후보 조회: fetch_modern_places_by_names 사용
    - 성경 본문 원문 조회: get_bible_passages 사용

    Args:
        keywords: 조회할 고대 성경 지명 이름 목록. 최대 5개까지 처리됩니다.
        top_k_per_keyword: 키워드별로 반환할 후보 수. 기본값 1, 1~2 사이로 제한됩니다.

    Returns:
        검색된 고대 지명의 place_id, 한글/영문 이름, 지명 유형,
        관련 성경 구절, 본문 내용. place_id 기준으로 중복은 제거됩니다.
    """
    normalized_keywords = [kw.strip() for kw in keywords if kw and kw.strip()]

    if not normalized_keywords:
        return []

    limited_keywords = normalized_keywords[:5]
    normalized_top_k = max(1, min(top_k_per_keyword, 2))

    embeddings = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=limited_keywords,
    ).data

    seen_ids: set[str] = set()
    results: list[PlaceSearchResult] = []

    for embedding_data in embeddings:
        result = places_index.query(
            vector=embedding_data.embedding,
            top_k=normalized_top_k,
            include_metadata=True,
            include_values=False,
            namespace="parent",
        )

        for match in result.get("matches", []):
            metadata = match["metadata"]
            place_id = metadata.get("place_id", match["id"])
            if place_id in seen_ids:
                continue
            seen_ids.add(place_id)
            results.append(
                PlaceSearchResult(
                    place_id=place_id,
                    name_ko=metadata.get("korean_name"),
                    name_en=metadata.get("name"),
                    types=metadata.get("types"),
                    bible_references=metadata.get("verses"),
                    identification_names=metadata.get("identification_names"),
                    text=metadata.get("text"),
                )
            )

    return results


@tool
def fetch_modern_places_by_names(
    names: list[str],
    top_k_per_name: int = 1,
) -> list[ModernPlaceRecord]:
    """현대 지명 이름 목록으로 관련 현대 지명 레코드를 검색합니다.

    각 이름을 임베딩하여 벡터 DB(child 네임스페이스)에서 가장 유사한
    현대 지명 레코드를 찾아 반환합니다.

    다음과 같은 상황에 사용합니다.
    - search_ancient_places 결과의 identification_names(고대 지명과 동일시되는
      현대 지명 후보들)로 실제 현대 지명 레코드를 조회할 때
    - 이미 알고 있는 현대 지명 이름들(예: "텔 키숀", "쿰란")의 상세 정보를
      한 번에 조회할 때

    다음 상황에는 사용하지 마세요.
    - 고대 성경 지명을 검색할 때: search_ancient_places 사용

    Args:
        names: 조회할 현대 지명 이름 목록. 최대 10개까지 처리됩니다.
        top_k_per_name: 이름별로 반환할 후보 수. 기본값 1, 1~2 사이로 제한됩니다.

    Returns:
        조회된 현대 지명의 place_id, 한글/영문 이름, 지명 유형,
        관련된 성경 장소 ID 목록, 본문 내용. place_id 기준으로 중복은 제거됩니다.
    """
    normalized_names = [name.strip() for name in names if name and name.strip()]

    if not normalized_names:
        return []

    limited_names = normalized_names[:10]
    normalized_top_k = max(1, min(top_k_per_name, 2))

    embeddings = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=limited_names,
    ).data

    seen_ids: set[str] = set()
    records: list[ModernPlaceRecord] = []

    for embedding_data in embeddings:
        result = places_index.query(
            vector=embedding_data.embedding,
            top_k=normalized_top_k,
            include_metadata=True,
            include_values=False,
            namespace="child",
        )

        for match in result.get("matches", []):
            metadata = match["metadata"]
            place_id = metadata.get("place_id", match["id"])
            if place_id in seen_ids:
                continue
            seen_ids.add(place_id)
            records.append(
                ModernPlaceRecord(
                    place_id=place_id,
                    name_ko=metadata.get("korean_name"),
                    name_en=metadata.get("name"),
                    types=metadata.get("types"),
                    parent_ids=metadata.get("parent_ids"),
                    text=metadata.get("text"),
                )
            )

    return records


@tool
def journey_search(query: str) -> list[str]:
    """
    성경 인물이나 집단의 이동 경로, 전도 여행 또는 여정의 순서를
    묻는 질문에 사용한다.

    질문에서 요구하는 여정의 장소들을 출발지부터 도착지까지
    실제 이동 순서대로 반환한다. 반환된 장소명 리스트는
    search_ancient_places의 keywords 인자로 그대로 넘겨
    각 지점의 상세 지명 데이터를 조회할 수 있다.
    """
    result = journey_search_chain.invoke({"query": query})
    return result.places


@tool
def ancient_keyword_search(query: str) -> list[str]:
    """
    사용자의 질문이 성경의 사건, 인물의 행동 또는 지리적 특징을 통해
    특정 고대 성경 장소를 묻는 경우 사용한다.

    질문의 단서를 분석하여 정답일 가능성이 높은 장소명을
    최대 3개까지 반환한다.

    일반적인 성경 지식, 신학적 개념 또는 지도에서 특정할 수 없는 장소를
    묻는 질문에는 사용하지 않는다.
    """
    result = keyword_search_chain.invoke({"query": query})
    return result.keywords


__all__ = [
    "PlaceSearchResult",
    "ModernPlaceRecord",
    "KeywordResult",
    "JourneyResult",
    "search_ancient_places",
    "fetch_modern_places_by_names",
    "journey_search",
    "ancient_keyword_search",
]
