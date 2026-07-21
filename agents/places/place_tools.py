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


class JourneyRouteResult(BaseModel):
    places: list[str] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "성경 인물이나 집단이 이동한 장소를 출발지부터 도착지까지 "
            "실제 이동 순서대로 나열한 목록. 여정이 아니면 빈 배열."
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


@tool
def search_ancient_places(
    keywords: list[str],
    top_k_per_keyword: int = 3,
) -> list[PlaceSearchResult]:
    """고대 성경 장소 이름 키워드 목록으로 지명 데이터를 벡터 검색합니다.

    각 키워드를 임베딩해 Pinecone places 인덱스의 parent 네임스페이스에서
    가장 유사한 고대 지명 레코드를 찾아 반환합니다.

    Args:
        keywords: 조회할 고대 성경 지명 이름 목록. 최대 5개까지 처리됩니다.
        top_k_per_keyword: 키워드별 후보 수. 기본값 3, 1~5 사이로 제한됩니다.

    Returns:
        고대 지명의 place_id, 한글/영문 이름, 지명 유형, 관련 성경 구절,
        현대 위치 후보 이름(identification_names), 본문 설명.
        place_id 기준으로 중복은 제거됩니다.
    """
    normalized_keywords = [kw.strip() for kw in keywords if kw and kw.strip()]

    if not normalized_keywords:
        return []

    limited_keywords = normalized_keywords[:5]
    normalized_top_k = max(1, min(top_k_per_keyword, 5))

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
    """현대 지명 이름 목록으로 현대 지명 레코드를 벡터 검색합니다.

    각 이름을 임베딩해 Pinecone places 인덱스의 child 네임스페이스에서
    가장 유사한 현대 지명 레코드를 찾아 반환합니다.

    Args:
        names: 조회할 현대 지명 이름 목록. 최대 10개까지 처리됩니다.
        top_k_per_name: 이름별 후보 수. 기본값 1, 1~2 사이로 제한됩니다.

    Returns:
        현대 지명의 place_id, 한글/영문 이름, 지명 유형,
        연관된 고대 성경 장소 ID 목록(parent_ids), 본문 설명.
        place_id 기준으로 중복은 제거됩니다.
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
    "KeywordResult",
    "JourneyRouteResult",
    "JourneyDescriptionResult",
    "search_ancient_places",
    "fetch_modern_places_by_names",
    "journey_route_search",
    "journey_description_search",
    "ancient_keyword_search",
]
