import json
from typing import Any, Dict, List, Literal, TypedDict

from langgraph.graph import END, START, StateGraph


SearchType = Literal["semantic", "keyword", "hybrid"]


class AgentState(TypedDict, total=False):
    question: str
    search_query: str
    search_type: SearchType
    top_k: int
    forced_top_k: int
    use_rerank: bool
    documents: List[Dict[str, Any]]
    answer: str


class LangGraphAgent:
    """Optional graph agent that orchestrates the existing RAG components."""

    def __init__(self, model_client, retriever, prompt_templates, max_candidates: int = 10):
        self.model = model_client
        self.retriever = retriever
        self.decision_prompt = prompt_templates["agent_decision_prompt"]
        self.qa_prompt = prompt_templates["qa_prompt"]
        self.rerank_prompt = prompt_templates["rerank_prompt"]
        self.max_candidates = max_candidates

        graph = StateGraph(AgentState)
        graph.add_node("decide_search", self.decide_search)
        graph.add_node("search", self.search)
        graph.add_node("rerank", self.rerank)
        graph.add_node("answer", self.answer_node)
        graph.add_edge(START, "decide_search")
        graph.add_edge("decide_search", "search")
        graph.add_conditional_edges(
            "search",
            self.route_after_search,
            {"rerank": "rerank", "answer": "answer"},
        )
        graph.add_edge("rerank", "answer")
        graph.add_edge("answer", END)
        self.graph = graph.compile()

    def decide_search(self, state: AgentState) -> AgentState:
        prompt = self.decision_prompt.format(question=state["question"])
        raw = self.model.generate(prompt, max_tokens=220)
        decision = self._parse_json(raw)
        search_type = decision.get("search_type", "hybrid")
        if search_type not in {"semantic", "keyword", "hybrid"}:
            search_type = "hybrid"
        top_k = state.get("forced_top_k", decision.get("top_k", 5))
        try:
            top_k = max(1, min(int(top_k), self.max_candidates))
        except (TypeError, ValueError):
            top_k = 5
        return {
            **state,
            "search_query": str(decision.get("query") or state["question"]),
            "search_type": search_type,
            "top_k": top_k,
            "use_rerank": bool(decision.get("use_rerank", False)),
        }

    def search(self, state: AgentState) -> AgentState:
        query = state["search_query"]
        top_k = state["top_k"]
        search_type = state["search_type"]
        if search_type == "semantic":
            documents = self.retriever.semantic_search(query, k=top_k)
        elif search_type == "keyword":
            documents = self.retriever.keyword_search(query, k=top_k)
        else:
            documents = self.retriever.hybrid(query, k=top_k)
        return {**state, "documents": documents}

    def route_after_search(self, state: AgentState) -> str:
        return "rerank" if state.get("use_rerank") and state.get("documents") else "answer"

    def rerank(self, state: AgentState) -> AgentState:
        documents = state.get("documents", [])
        candidates = "\n\n".join(
            f"[{index}] {document.get('text', '')}"
            for index, document in enumerate(documents)
        )
        prompt = self.rerank_prompt.format(
            question=state["question"],
            candidates=candidates,
        )
        raw = self.model.generate(prompt, max_tokens=220)
        ranking = self._parse_json(raw).get("ranking", [])
        ordered = []
        seen = set()
        for value in ranking:
            try:
                index = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(documents) and index not in seen:
                ordered.append(documents[index])
                seen.add(index)
        ordered.extend(document for index, document in enumerate(documents) if index not in seen)
        return {**state, "documents": ordered[: state["top_k"]]}

    def answer_node(self, state: AgentState) -> AgentState:
        context = "\n\n".join(
            f"Source: {document.get('source', 'unknown')}\nText: {document.get('text', '')}"
            for document in state.get("documents", [])
        )
        prompt = self.qa_prompt.format(context=context, question=state["question"])
        answer = self.model.generate(prompt).strip()
        return {**state, "answer": answer or "I could not find the answer in the provided context."}

    def run(self, question: str, top_k: int | None = None) -> Dict[str, Any]:
        initial_state: AgentState = {"question": question}
        if top_k is not None:
            initial_state["forced_top_k"] = top_k
        state = self.graph.invoke(initial_state)
        return {
            "answer": state.get("answer", "I could not find the answer in the provided context."),
            "retrieved_documents": state.get("documents", []),
            "search_query": state.get("search_query"),
            "search_type": state.get("search_type"),
            "top_k": state.get("top_k"),
            "use_rerank": state.get("use_rerank", False),
            "sources": [
                {
                    "source": document.get("source"),
                    "page": document.get("page"),
                    "slide": document.get("slide"),
                    "chunk_id": document.get("chunk_id"),
                    "backend": document.get("backend"),
                    "score": document.get("score"),
                }
                for document in state.get("documents", [])
            ],
        }

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        try:
            value = json.loads(raw)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                try:
                    value = json.loads(raw[start : end + 1])
                    return value if isinstance(value, dict) else {}
                except json.JSONDecodeError:
                    pass
        return {}