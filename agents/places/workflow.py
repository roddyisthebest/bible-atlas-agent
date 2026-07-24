"""Place agent workflow graph for LangGraph Platform deployment.

Exports `graph` at module level. Deploy target for LangGraph Platform is
`agents/places/workflow.py:graph`.

The runtime layout:

    START
      │
   [router]  (conditional)
      ├─→ place_agent
      │     │
      │  [route_from_agent] (conditional)
      │     ├─→ tools ─→ place_agent (loop)
      │     ├─→ rewrite ─→ END
      │     └─→ format  ─→ END
      ├─→ bible_general_agent ─→ END
      └─→ non_bible_reject    ─→ END
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

from agents.places.place_tools import (
    ancient_keyword_search,
    journey_description_search,
    journey_route_search,
    search_ancient_places,
    search_ancient_with_modern,
    search_modern_places,
    search_modern_with_ancient,
)

load_dotenv()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class AgentState(MessagesState):
    """Agent workflow state.

    - query: 원본 사용자 질문 (실행 내내 불변)
    - answer: 최종 답변 텍스트
    - place_id_map: 답변에 언급될 수 있는 지명 → place_id 매핑 (프론트 매칭용)
    - recommended_questions: non_bible_reject 시 대안 예시 질문
    - messages: 대화·도구 호출 로그 (place_agent가 사용)
    """

    query: str
    answer: str
    # 이름 → [place_id, ...]. 같은 이름이 여러 record 에 걸릴 수 있어 리스트.
    # 프론트는 id 첫 글자로 stereo 판별 (a... = ancient, m... = modern).
    place_id_map: dict[str, list[str]]
    recommended_questions: list[str]


# ---------------------------------------------------------------------------
# LLMs
# ---------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o")
small_llm = ChatOpenAI(model="gpt-4o-mini")
mini_llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=400, temperature=0.2)
nano_llm = ChatOpenAI(model="gpt-4.1-nano", max_tokens=300, temperature=0.3)
judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
rewrite_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=500)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
# create_agent 가 내부적으로 llm.bind_tools + tool loop 를 처리하므로
# 별도의 llm_with_tools / tool_node 는 만들지 않는다.
tool_list = [
    ancient_keyword_search,
    search_ancient_places,
    search_modern_places,
    search_ancient_with_modern,
    search_modern_with_ancient,
    journey_route_search,
    journey_description_search,
]


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
class Route(BaseModel):
    target: Literal["place_agent", "bible_general_agent", "non_bible_reject"] = Field(
        description=(
            "사용자 질문을 분류할 대상 노드. "
            "성경의 지명·지리·여정에 관한 질문은 'place_agent', "
            "성경 관련이지만 장소가 초점이 아닌 질문(인물·교리·사건·해석 등)은 'bible_general_agent', "
            "성경과 무관한 질문은 'non_bible_reject'로 분류한다."
        )
    )


router_system_prompt = """
당신은 사용자 질문을 아래 세 target 중 하나로 분류하는 라우터입니다.
target은 반드시 'place_agent', 'bible_general_agent', 'non_bible_reject' 중 하나여야 합니다.

- 'place_agent': 질문이 성경의 지리·장소에 관한 경우.
  성경 지명(고대/현대), 인물·집단의 이동 경로·여정, 지역의 위치·의미,
  성경 지역과 현대 위치의 관계 등을 묻는 질문이 여기에 해당한다.
- 'bible_general_agent': 질문이 성경과 관련 있지만 장소가 주된 초점이 아닌 경우.
  성경 인물의 성품·행적, 교리·신학, 사건의 의미, 구절 해석,
  신앙적 개념, 성경적 지침 등이 여기에 해당한다.
- 'non_bible_reject': 질문이 성경 도메인과 무관한 경우.
  일상 잡담, 시사, 기술, 요리, 날씨, 취미 등 성경과 관련 없는
  모든 질문이 여기에 해당한다.

판단 규칙:
- 질문에 성경 지명이 명시적으로 등장하거나 인물·집단의 '이동/여정'을 묻는다면 'place_agent'.
- 성경에 등장하는 개념·인물·구절·교리를 묻지만 장소 이동이 초점이 아니라면 'bible_general_agent'.
- 성경적 요소가 전혀 없다면 'non_bible_reject'.
- 애매하면 성경 도메인 쪽으로 우선 분류하고, 장소가 조금이라도 관련되면 'place_agent'를 선택한다.

target만 반환하고 다른 텍스트는 출력하지 마세요.
"""

router_prompt = ChatPromptTemplate.from_messages([
    ("system", router_system_prompt),
    ("user", "{query}"),
])

router_chain = router_prompt | small_llm.with_structured_output(Route)


def router(state: AgentState) -> str:
    return router_chain.invoke({"query": state["query"]}).target


# ---------------------------------------------------------------------------
# place_agent
# ---------------------------------------------------------------------------
# PRD (docs/agent-prd.md) 가 시스템 프롬프트의 canonical source.
# workflow.py는 agents/places/에 있으므로 parent.parent.parent = 프로젝트 루트.
_PRD_PATH = Path(__file__).resolve().parent.parent.parent / "docs" / "agent-prd.md"
PLACE_AGENT_SYSTEM = _PRD_PATH.read_text(encoding="utf-8")

# create_agent 가 tool 호출 loop 를 내부에서 처리 (LangGraph 최적화된 ReAct 구현).
_place_agent_runnable = create_agent(
    model=llm,
    tools=tool_list,
    system_prompt=PLACE_AGENT_SYSTEM,
)


def place_agent(state: AgentState) -> dict:
    """tool loop 는 내부에서 완료. 새로 추가된 messages 만 delta 로 반환."""
    initial_count = len(state["messages"])
    result = _place_agent_runnable.invoke({"messages": state["messages"]})
    return {"messages": result["messages"][initial_count:]}


# ---------------------------------------------------------------------------
# bible_general_agent
# ---------------------------------------------------------------------------
bible_general_agent_prompt = PromptTemplate.from_template(
    """
    당신은 성경 질문에 답하는 도우미입니다.
    아래 규칙을 반드시 지키세요.

    규칙:
    - 3~5문장 이내로 간결하게 답합니다.
    - 성경 본문·인물·사건·신학 주제에 한정된 답변만 합니다.
    - 확신할 수 없는 부분은 추측하지 말고 "성경에서 명확히 다루지 않는다"고 답합니다.
    - 불필요한 서두·맺음말·이모지는 사용하지 않습니다.

    질문: {query}
    """
)

bible_general_agent_chain = bible_general_agent_prompt | mini_llm | StrOutputParser()


def bible_general_agent(state: AgentState) -> dict:
    return {"answer": bible_general_agent_chain.invoke({"query": state["query"]})}


# ---------------------------------------------------------------------------
# non_bible_reject
# ---------------------------------------------------------------------------
class NonBibleResponse(BaseModel):
    answer: str = Field(
        description=(
            "질문이 성경 도메인과 무관하다는 안내 메시지. 1~2문장. 정중하고 짧게."
        )
    )
    recommended_questions: list[str] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "사용자가 대신 물어볼 만한 성경 지리·지명 관련 예시 질문 3개. "
            "가능하면 사용자의 원 질문 맥락(주제·인물·상황)과 연결지어 유도. "
            "질문 형태로 작성."
        ),
    )


non_bible_reject_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        사용자의 질문이 성경 도메인과 무관합니다.
        아래 두 필드를 채워 반환하세요.

        - answer: 1~2문장. "성경과 무관한 질문이라 답변이 어렵다"는 취지의 정중한 안내.
        - recommended_questions: 사용자가 대신 물어볼 만한 성경 지리·지명 관련
          예시 질문 3개. 원 질문의 주제·인물·상황과 조금이라도 연결 가능하면
          그 쪽으로 유도.

        이모지·부연 설명은 넣지 마세요.
        """,
    ),
    ("user", "질문: {query}"),
])

non_bible_reject_chain = (
    non_bible_reject_prompt | nano_llm.with_structured_output(NonBibleResponse)
)


def non_bible_reject(state: AgentState) -> dict:
    result = non_bible_reject_chain.invoke({"query": state["query"]})
    return {
        "answer": result.answer,
        "recommended_questions": result.recommended_questions,
    }


# ---------------------------------------------------------------------------
# evaluate_answer (routing decision after agent)
# ---------------------------------------------------------------------------
class RelevanceCheck(BaseModel):
    is_relevant: bool = Field(
        description="답변이 질문에 실제로 부합하고 사실적으로 문제가 없으면 true, 그렇지 않으면 false"
    )
    reason: str = Field(default="", description="판단 근거를 한 문장 이내로 짧게 기술")


relevance_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        당신은 답변이 질문에 얼마나 잘 부합하는지 평가하는 판정자입니다.
        아래 기준으로만 판단하세요.

        기준:
        - 답변이 질문의 요지를 정면으로 다루고 있는가.
        - 명백한 사실적 오류가 있는가.
        - 질문과 무관한 내용으로 회피하고 있는가.
        - 답변이 비어 있거나 지나치게 모호한가.

        기준을 모두 만족하면 is_relevant=true, 하나라도 어긋나면 false.
        간결하게 판단하고 부연 설명은 하지 마세요.
        """,
    ),
    ("user", "질문: {query}\n답변: {answer}"),
])

relevance_chain = relevance_prompt | judge_llm.with_structured_output(RelevanceCheck)


def evaluate_answer(state: AgentState) -> Literal["rewrite", "format"]:
    check = relevance_chain.invoke({
        "query": state["query"],
        "answer": state["messages"][-1].content,
    })
    return "rewrite" if not check.is_relevant else "format"


# ---------------------------------------------------------------------------
# rewrite (LLM 자체 지식 기반 fallback 답변)
# ---------------------------------------------------------------------------
rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        당신은 성경 지리·지명 전문가입니다.
        데이터베이스 검색으로 충분한 정보를 얻지 못해, 이번 답변은
        당신의 성경 사전 지식에만 의존해 작성해야 합니다.

        지침:
        - 성경 본문·전통·널리 알려진 학설 수준의 정보로 답합니다.
        - 확실히 알려진 사실은 단정적으로 답합니다.
        - 학자 간 이견이 있는 부분은 "학자들 사이에 의견이 갈리지만 흔히…" 같은 표현을 붙입니다.
        - 정확한 좌표, place_id, 확신할 수 없는 세부 수치는 답하지 마세요.
        - 성경이 명시하지 않은 사항은 "성경에서 명확히 다루지 않음"이라고 답합니다.
        - 3~5문장 이내로 간결하게 작성합니다.
        - 사과·메타 발언·이모지는 넣지 않습니다.
        """,
    ),
    ("user", "질문: {query}"),
])

rewrite_chain = rewrite_prompt | rewrite_llm | StrOutputParser()


def rewrite(state: AgentState) -> dict:
    return {"answer": rewrite_chain.invoke({"query": state["query"]})}


# ---------------------------------------------------------------------------
# format (place_id 매핑 추출 + 답변 확정)
# ---------------------------------------------------------------------------
def _extract_place_refs(tool_message_content: str) -> dict[str, list[str]]:
    """도구 결과 JSON 에서 {이름: [place_id, ...]} 매핑을 뽑는다.

    - top-level record + nested modern_places / ancient_places 재귀 순회.
    - 같은 이름에 여러 place_id 가 붙을 수 있음 (부모/자식 동일 영문명 등).
    - id 첫 글자로 stereo 판별 가능 (a... = ancient, m... = modern).
    """
    refs: dict[str, list[str]] = {}

    def _add(name: str, pid: str) -> None:
        bucket = refs.setdefault(name, [])
        if pid not in bucket:
            bucket.append(pid)

    def _walk(item):
        if not isinstance(item, dict):
            return
        pid = item.get("place_id")
        if pid:
            for key in ("name_ko", "name_en"):
                name = item.get(key)
                if name:
                    _add(name, pid)
        for nested_key in ("modern_places", "ancient_places"):
            for nested_item in item.get(nested_key) or []:
                _walk(nested_item)

    try:
        data = json.loads(tool_message_content)
    except (TypeError, ValueError):
        return refs
    if not isinstance(data, list):
        return refs
    for item in data:
        _walk(item)
    return refs


def _merge_refs(dst: dict[str, list[str]], src: dict[str, list[str]]) -> None:
    for name, ids in src.items():
        bucket = dst.setdefault(name, [])
        for pid in ids:
            if pid not in bucket:
                bucket.append(pid)


def format(state: AgentState) -> dict:
    messages = state["messages"]
    # rewrite 에서 answer 이미 세팅됐으면 그것 보존, 아니면 마지막 메시지 사용
    answer_text = state.get("answer") or messages[-1].content

    place_id_map: dict[str, list[str]] = {}
    for msg in messages:
        if isinstance(msg, ToolMessage):
            _merge_refs(place_id_map, _extract_place_refs(msg.content))

    return {"answer": answer_text, "place_id_map": place_id_map}


# ---------------------------------------------------------------------------
# route_from_agent (conditional edge)
# ---------------------------------------------------------------------------
def route_from_agent(state: AgentState) -> Literal["rewrite", "format"]:
    """tool loop 는 create_agent 내부에서 완료됨. 여기선 답변 평가만 분기."""
    return evaluate_answer(state)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
graph_builder = StateGraph(AgentState)

graph_builder.add_node("place_agent", place_agent)
graph_builder.add_node("bible_general_agent", bible_general_agent)
graph_builder.add_node("non_bible_reject", non_bible_reject)
graph_builder.add_node("rewrite", rewrite)
graph_builder.add_node("format", format)

graph_builder.add_conditional_edges(START, router, {
    "place_agent": "place_agent",
    "bible_general_agent": "bible_general_agent",
    "non_bible_reject": "non_bible_reject",
})

graph_builder.add_conditional_edges("place_agent", route_from_agent, {
    "rewrite": "rewrite",
    "format": "format",
})

# rewrite 도 format 을 거쳐서 place_id_map 을 채우고 종료
graph_builder.add_edge("rewrite", "format")

graph_builder.add_edge("bible_general_agent", END)
graph_builder.add_edge("non_bible_reject", END)
graph_builder.add_edge("format", END)

graph = graph_builder.compile()


__all__ = [
    "AgentState",
    "graph",
    "PLACE_AGENT_SYSTEM",
    "Route",
    "NonBibleResponse",
    "RelevanceCheck",
]
