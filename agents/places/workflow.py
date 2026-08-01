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
import re
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
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
            "사용자 질문을 분류할 대상 노드입니다. "
            "성경의 지명·지리·여정에 관한 질문은 'place_agent', "
            "성경 관련이지만 장소가 초점이 아닌 질문(인물·교리·사건·해석 등)은 'bible_general_agent', "
            "성경과 무관한 질문은 'non_bible_reject' 로 분류합니다."
        )
    )


router_system_prompt = """사용자 질문을 세 target 중 하나로 분류:
- `place_agent`: 성경 지명·지리·여정·현대 위치 관계
- `bible_general_agent`: 성경 인물·교리·구절 해석·개념 (장소 중심 아님)
- `non_bible_reject`: 성경 무관

규칙:
- 성경 지명·여정 명시 → place_agent
- 성경 개념·인물·구절인데 장소 초점 아님 → bible_general_agent
- 성경 요소 없음 → non_bible_reject
- 애매하면 성경 쪽, 장소 조금이라도 관련되면 place_agent

**context 활용**: 현재 질문이 지시대명사(그곳/거기/그때 등)나 후속 표현(그럼~/이후에 등)을 포함하고 [context]가 성경 지리면 place_agent, 성경 인물·교리면 bible_general_agent 로 유지. context 가 "(없음)"이고 현재 질문에 성경 신호 없으면 non_bible_reject. 이전과 무관한 새 주제면 현재 질문 기준.
"""

router_prompt = ChatPromptTemplate.from_messages([
    ("system", router_system_prompt),
    ("user", "[context]\n{context}\n\n[현재 질문]\n{query}"),
])

router_chain = router_prompt | small_llm.with_structured_output(Route)


def _build_router_context(messages: list) -> str:
    """Compact context for router: summary (if any) + last ~2 exchanges.

    messages 구조 (chat.build_context 결과):
        [SystemMessage(summary)?, HumanMessage(q1), AIMessage(a1), ...,
         HumanMessage(current_query)]
    현재 질문(last HumanMessage)은 별도 slot 으로 넘기므로 여기서 제외.
    """
    if len(messages) <= 1:
        return "(없음)"

    lines: list[str] = []
    for msg in messages[:-1]:
        if isinstance(msg, SystemMessage):
            lines.append(msg.content)
        elif isinstance(msg, HumanMessage):
            lines.append(f"user: {msg.content[:300]}")
        elif isinstance(msg, AIMessage):
            lines.append(f"assistant: {msg.content[:300]}")

    if not lines:
        return "(없음)"
    # 요약 있으면 그대로 유지, 그 뒤로 최근 대화 최대 4줄만.
    return "\n".join(lines[-5:])


def router(state: AgentState) -> str:
    context = _build_router_context(state.get("messages") or [])
    return router_chain.invoke({
        "query": state["query"],
        "context": context,
    }).target


# ---------------------------------------------------------------------------
# Shared prompt fragments
# ---------------------------------------------------------------------------
# 모든 answer 생성 노드에 공통 주입되는 규칙 세트. 각 프롬프트 뒤에
# concat 해서 언어/문단 형식을 강제한다.
_LANGUAGE_RULE = (
    "**언어 규칙**\n"
    "- 사용자 질문 언어와 반드시 동일한 언어로 답변합니다. "
    "한국어 질문 → 한국어, 영어 질문 → 영어. 다른 언어로 답하지 않습니다.\n"
    "- 한국어일 때만 존댓말(-습니다체)을 사용합니다. 반말·평서체(-다) 금지.\n"
    "- 영어일 때는 정중한(polite) 톤을 유지합니다."
)

# 프론트가 문단 단위로 렌더할 수 있도록 논리 단위 사이를 \n\n 으로 분리.
_FORMAT_RULE = (
    "**출력 형식 규칙**\n"
    "- 답변이 논리적으로 둘 이상의 단위로 나뉘면 각 단위 사이는 반드시 "
    "개행 두 번(`\\n\\n`)으로 문단을 분리하세요.\n"
    "- 한 문단으로 충분한 짧은 답은 그대로 한 문단으로 유지합니다.\n"
    "- 문단 내부에서 임의 줄바꿈은 사용하지 않습니다."
)

# 각 프롬프트에 한 번에 붙일 수 있게 미리 조립.
_COMMON_ANSWER_RULES = _LANGUAGE_RULE + "\n\n" + _FORMAT_RULE


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
    system_prompt=PLACE_AGENT_SYSTEM + "\n\n" + _COMMON_ANSWER_RULES,
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
    """성경 질문 응답 도우미.
규칙:
- 3~5문장 이내, 성경 본문·인물·사건·신학에 한정.
- 확신 없으면 "성경에서 명확히 다루지 않는다" (영어면 "the Bible does not clearly address this").
- 서두·맺음말·이모지 금지.

""" + _COMMON_ANSWER_RULES + """

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
            "Polite off-topic reply. 3~5 sentences. MUST include `\\n\\n` "
            "between logical paragraphs. Language MUST match the user's "
            "query language (Korean query -> Korean answer, English query "
            "-> English answer)."
        )
    )
    recommended_questions: list[str] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "Exactly 3 example biblical-geography questions. Language MUST "
            "match the user's query language."
        ),
    )


non_bible_reject_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        _COMMON_ANSWER_RULES + """

You are a specialist assistant for **biblical places, geography, and journeys**.
The user's current question is outside this domain. Fill the two fields below and return.

**CRITICAL: The [Language Rule] above is absolute.** Detect the user's query language
and write EVERY field (answer + every recommended_questions entry) in that same
language. Korean query → all Korean output. English query → all English output.
Do NOT default to Korean. Do NOT mix languages.

[answer content]
- Exactly 3–5 sentences, structured as three parts separated by `\\n\\n`:
    1) One sentence that empathizes with the user's intent.
    2) One sentence explaining this service is specialized for biblical
       places / geography / journeys and cannot address this topic in detail.
    3) One sentence pivoting toward biblical-geography topics we can help with.
- No repeated apologies, no emojis, no overstatement.

[recommended_questions content]
- Exactly 3 example questions about biblical places or geography.
- If the original question has any thread (region, person, era, theme) that can
  bridge into biblical geography, bias the examples toward that thread.
- Each example question MUST be in the same language as the user's query.

[Language detection examples — follow this pattern strictly]
- Query "What's your favorite movie?" → detect language = English → answer in English, recommended_questions in English.
- Query "where is the money" → English → English output.
- Query "오늘 날씨 어때?" → Korean → Korean output.
- Query "돈 어디있어" → Korean → Korean output.
Detect the query language BEFORE writing anything, then commit to that language for every field.""",
    ),
    ("user", "User query (detect language, respond in that same language): {query}"),
])

# nano 모델은 다국어 + 문단 형식 같은 복합 지시를 잘 못 따라가서
# mini 로 승급. 구조화 출력이라 비용 여전히 작음.
non_bible_reject_chain = (
    non_bible_reject_prompt | mini_llm.with_structured_output(NonBibleResponse)
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
        """당신은 답변이 사용자 질문과 도구 검색 결과에 얼마나 잘 부합하는지
평가하는 판정자입니다.

판단 기준:
- 답변이 질문의 요지를 정면으로 다루는가.
- 도구 결과가 있으면 답변이 그 결과에 근거하는가.
- 서로 다른 장소를 하나로 합치거나, 도구 결과에 없는 place_id/좌표/현대
  지명을 만들어내지 않았는가.
- 명백한 사실 오류나 회피성 답변이 아닌가.

**아래 조건 중 하나라도 해당하면 반드시 is_relevant = false:**
1. 도구 결과가 전부 비어있고([] 또는 빈 문자열), 답변에
   "정보가 없다", "명확한 지명이 제공되지 않았", "구체적 자료가 없",
   "다른 자료를 참고", "명시되지 않았", "제공되지 않았"
   같은 정보 부족을 인정하는 표현이 있는 경우.
   → rewrite 노드가 LLM 사전 지식으로 더 나은 답변을 만들어야 함.
2. 답변이 질문의 핵심(지명·인물·사건)을 다루지 못하고 회피.
3. 답변이 비어있거나 지나치게 모호.

**예외 - 다음은 is_relevant = true 허용:**
- 질문 자체가 성경 밖 초월적/추상적 주제(예: "천국은 어디?")이고
  답변이 성경적 관점에서 적절히 짚어준 경우.
- 도구 결과가 존재하고 답변이 그 결과와 일관되게 작성된 경우.

reason에는 판단 근거 한 문장만 짧게.""",
    ),
    (
        "user",
        """질문:
{query}

도구 검색 결과:
{tool_results}

최종 답변:
{answer}""",
    ),
])

relevance_chain = relevance_prompt | judge_llm.with_structured_output(RelevanceCheck)


def evaluate_answer(state: AgentState) -> Literal["rewrite", "format"]:
    tool_results = [
        message.content
        for message in state["messages"]
        if isinstance(message, ToolMessage)
    ]
    check = relevance_chain.invoke({
        "query": state["query"],
        "answer": state["messages"][-1].content,
        "tool_results": tool_results,
    })
    return "rewrite" if not check.is_relevant else "format"


# ---------------------------------------------------------------------------
# rewrite (LLM 자체 지식 기반 fallback 답변)
# ---------------------------------------------------------------------------
rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """성경 지리·지명 전문가. DB 결과 부족해 사전 지식만으로 답.
- 성경 본문·전통·널리 알려진 학설 수준.
- 확실한 사실은 단정, 이견 있으면 "학자들 사이 의견이 갈리지만 흔히…" (영어면 "scholars are divided, but commonly…") 첨언.
- 좌표·place_id·확신 없는 수치 금지.
- 성경 미명시는 "성경에서 명확히 다루지 않음" (영어면 "the Bible does not clearly address this").
- 3~5문장. 사과·메타·이모지 금지.

""" + _COMMON_ANSWER_RULES,
    ),
    ("user", "질문: {query}"),
])

rewrite_chain = rewrite_prompt | rewrite_llm | StrOutputParser()


def rewrite(state: AgentState) -> dict:
    return {"answer": rewrite_chain.invoke({"query": state["query"]})}


# ---------------------------------------------------------------------------
# format (place_id 매핑 추출 + 답변 확정)
# ---------------------------------------------------------------------------
_TRAILING_DIGITS_RE = re.compile(r"\s*\d+$")
_TRAILING_PAREN_RE = re.compile(r"\s*\([^)]*\)\s*$")


def _normalize_place_name(name: str) -> str:
    """DB 동명이인 구분자(접미 숫자 + 괄호 주석)를 제거해 base name 을 만든다.

    예)
        "안디옥1"                → "안디옥"
        "안디옥 2"               → "안디옥"
        "Antioch3"              → "Antioch"
        "베들레헴 1 (유다)"       → "베들레헴"
        "베들레헴 (스블론)"       → "베들레헴"
        "Bethlehem 2 (Zebulun)" → "Bethlehem"

    이렇게 하면 (a) 고대·현대가 같은 이름을 가질 때, (b) DB 가 숫자·괄호로
    동명을 구분해 저장했을 때, place_id_map 이 base name 하나의 key 로
    모든 후보 id 를 버킷팅한다. 결과가 빈 문자열이면 원본 유지.
    """
    prev = None
    curr = name
    # 괄호 주석과 숫자 접미가 어느 순서로 붙어 있어도 반복해서 벗겨낸다.
    while prev != curr:
        prev = curr
        curr = _TRAILING_PAREN_RE.sub("", curr).rstrip()
        curr = _TRAILING_DIGITS_RE.sub("", curr).rstrip()
    return curr or name


def _extract_place_refs(tool_message_content: str) -> dict[str, list[str]]:
    """도구 결과 JSON 에서 {이름: [place_id, ...]} 매핑을 뽑는다.

    - top-level record + nested modern_places / ancient_places 재귀 순회.
    - 같은 이름(정규화 후)에 여러 place_id 가 붙을 수 있음
      (부모/자식 동일명, 동명이인 접미 숫자 등).
    - id 첫 글자로 stereo 판별 가능 (a... = ancient, m... = modern).
    """
    refs: dict[str, list[str]] = {}

    def _add(name: str, pid: str) -> None:
        key = _normalize_place_name(name)
        bucket = refs.setdefault(key, [])
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


_PLACE_ID_MAP_BLOCK_RE = re.compile(
    r"\n+\s*\**\s*place[_ ]?id[_ ]?map\s*\**\s*:.*\Z",
    re.IGNORECASE | re.DOTALL,
)


def _strip_place_id_map_block(text: str) -> str:
    """답변 말미에 LLM 이 place_id_map 을 markdown 으로 다시 나열하는 경우
    (PRD Rule 11 위반) 를 방어적으로 잘라낸다. 프롬프트가 실패해도 프론트로
    는 절대 나가지 않게 하는 최후 방어선."""
    return _PLACE_ID_MAP_BLOCK_RE.sub("", text).rstrip()


def format(state: AgentState) -> dict:
    messages = state["messages"]
    # rewrite 에서 answer 이미 세팅됐으면 그것 보존, 아니면 마지막 메시지 사용
    answer_text = state.get("answer") or messages[-1].content
    answer_text = _strip_place_id_map_block(answer_text)

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
