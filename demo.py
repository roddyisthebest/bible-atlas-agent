import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
)


class AgentState(TypedDict):
    context: list
    query: str
    answer: str


def generate_answer(state: AgentState) -> dict:
    context = state["context"]
    query = state["query"]

    response = llm.invoke(
        f"""
        다음 Context를 참고해 질문에 답변하세요.

        Context:
        {context}

        Question:
        {query}
        """
    )

    return {"answer": response.content}


graph_builder = StateGraph(AgentState)
graph_builder.add_node("generate_answer", generate_answer)
graph_builder.add_edge(START, "generate_answer")
graph_builder.add_edge("generate_answer", END)

graph = graph_builder.compile()


if __name__ == "__main__":
    initial_state: AgentState = {
        "context": [],
        "query": "아가야는 어떤 곳이야?",
        "answer": "",
    }

    final_state = graph.invoke(initial_state)
    print(final_state["answer"])
