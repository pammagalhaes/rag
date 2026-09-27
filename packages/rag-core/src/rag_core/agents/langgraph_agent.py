import json
from typing import Any, Dict, List, Literal, TypedDict
from contextlib import nullcontext

from langgraph.graph import END, START, StateGraph


SearchType = Literal["semantic", "keyword", "hybrid"]


class AgentState(TypedDict, total=False):
    question: str
    search_query: str
    search_type: SearchType
    top_k: int
    candidate_k: int
    forced_top_k: int
    use_rerank: bool
    documents: List[Dict[str, Any]]
    candidate_count_after_search: int
    candidate_count_after_rerank: int
    final_count: int
    answer: str


class LangGraphAgent:
    """Optional graph agent that orchestrates the existing RAG components."""

    def __init__(
        self,
        model_client,
        retriever,
        prompt_templates,
        max_candidates: int = 10,
        protect_baseline: bool = True,
        rerank_enabled: bool = True,
        tracer=None,
    ):
        self.model = model_client
        self.retriever = retriever
        self.decision_prompt = prompt_templates["agent_decision_prompt"]
        self.qa_prompt = prompt_templates["qa_prompt"]
        self.rerank_prompt = prompt_templates["rerank_prompt"]
        self.max_candidates = max_candidates
        self.protect_baseline = protect_baseline
        self.rerank_enabled = rerank_enabled
        self.tracer = tracer

        graph = StateGraph(AgentState)
        graph.add_node("decide_search", self._wrap_node("agent.decide_search", self.decide_search))
        graph.add_node("search", self._wrap_node("agent.search", self.search))
        graph.add_node("rerank", self._wrap_node("agent.rerank", self.rerank))
        graph.add_node("answer", self._wrap_node("agent.answer", self.answer_node))
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
        candidate_k = max(top_k, self.max_candidates)
        rerank_value = decision.get("use_rerank", False)
        if isinstance(rerank_value, str):
            rerank_value = rerank_value.strip().lower() in {"true", "1", "yes"}
        return {
            **state,
            "search_query": str(decision.get("query") or state["question"]),
            "search_type": search_type,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "use_rerank": bool(rerank_value) and self.rerank_enabled,
        }

    def search(self, state: AgentState) -> AgentState:
        query = state["search_query"]
        top_k = state["candidate_k"]
        search_type = state["search_type"]
        baseline_documents = (
            self.retriever.hybrid(state["question"], k=top_k)
            if self.protect_baseline
            else []
        )
        if search_type == "semantic":
            documents = self.retriever.semantic_search(query, k=top_k)
        elif search_type == "keyword":
            documents = self.retriever.keyword_search(query, k=top_k)
        else:
            documents = self.retriever.hybrid(query, k=top_k)

        if not documents:
            fallback_query = state["question"]
            fallback_searches = [
                ("hybrid_fallback", lambda: self.retriever.hybrid(fallback_query, k=top_k)),
                ("semantic_fallback", lambda: self.retriever.semantic_search(fallback_query, k=top_k)),
                ("keyword_fallback", lambda: self.retriever.keyword_search(fallback_query, k=top_k)),
            ]
            for fallback_type, fallback_documents in fallback_searches:
                documents = fallback_documents()
                if documents:
                    search_type = fallback_type
                    break

        merged_documents = []
        seen = set()
        for document in [*baseline_documents, *documents]:
            document_key = document.get("chunk_id") or (
                document.get("source"), document.get("text", "")[:100]
            )
            if document_key not in seen:
                merged_documents.append(document)
                seen.add(document_key)

        candidate_documents = merged_documents[: state["candidate_k"]]
        return {
            **state,
            "documents": candidate_documents,
            "candidate_count_after_search": len(candidate_documents),
            "search_type": search_type,
        }

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
        reranked_documents = ordered[: state["top_k"]]
        return {
            **state,
            "documents": reranked_documents,
            "candidate_count_after_rerank": len(reranked_documents),
        }

    def answer_node(self, state: AgentState) -> AgentState:
        documents = state.get("documents", [])[: state["top_k"]]
        context = "\n\n".join(
            f"Source: {document.get('source', 'unknown')}\nText: {document.get('text', '')}"
            for document in documents
        )
        prompt = self.qa_prompt.format(context=context, question=state["question"])
        answer = self.model.generate(prompt).strip()
        return {
            **state,
            "documents": documents,
            "final_count": len(documents),
            "answer": answer or "I could not find the answer in the provided context.",
        }

    def run(
        self,
        question: str,
        top_k: int | None = None,
        trace_metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        initial_state: AgentState = {"question": question}
        if top_k is not None:
            initial_state["forced_top_k"] = top_k
        observation_context = (
            self.tracer.observation(
                "langgraph.agent",
                input_data=question,
                metadata={
                    **(trace_metadata or {}),
                    "top_k_requested": top_k,
                    "max_candidates": self.max_candidates,
                    "protect_baseline": self.protect_baseline,
                    "rerank_enabled": self.rerank_enabled,
                },
            )
            if self.tracer is not None
            else nullcontext(None)
        )
        trace_id = None
        with observation_context as trace:
            if trace is not None:
                trace_id = trace.trace_id
            state = self.graph.invoke(initial_state)
            if trace is not None:
                trace.update(
                    metadata=self._node_metadata("agent.result", state),
                    **(
                        {"output": state.get("answer", "")}
                        if self.tracer.capture_content
                        else {}
                    ),
                )
        return {
            "langfuse_trace_id": trace_id,
            "answer": state.get("answer", "I could not find the answer in the provided context."),
            "retrieved_documents": state.get("documents", []),
            "search_query": state.get("search_query"),
            "search_type": state.get("search_type"),
            "top_k": state.get("top_k"),
            "use_rerank": state.get("use_rerank", False),
            "candidate_count_after_search": state.get("candidate_count_after_search", 0),
            "candidate_count_after_rerank": state.get(
                "candidate_count_after_rerank", 0
            ),
            "final_count": state.get("final_count", 0),
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

    def _wrap_node(self, name: str, node):
        def traced_node(state: AgentState) -> AgentState:
            if self.tracer is None:
                return node(state)
            with self.tracer.observation(
                name,
                input_data={"question": state.get("question")},
                metadata={"node": name.rsplit(".", 1)[-1]},
            ) as observation:
                result = node(state)
                observation.update(
                    metadata=self._node_metadata(name, result),
                    **(
                        {"output": result.get("answer", "")}
                        if self.tracer.capture_content and name.endswith("answer")
                        else {}
                    ),
                )
                return result

        return traced_node

    def _node_metadata(self, name: str, state: AgentState) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        for key in (
            "search_type",
            "top_k",
            "candidate_k",
            "use_rerank",
            "candidate_count_after_search",
            "candidate_count_after_rerank",
            "final_count",
        ):
            if key in state:
                metadata[key] = state[key]
        if self.tracer is not None and self.tracer.capture_content and "search_query" in state:
            metadata["search_query"] = state["search_query"]
        documents = state.get("documents", [])
        if documents:
            metadata["documents"] = [
                self._document_reference(document)
                for document in documents
            ]
        metadata["stage"] = name.rsplit(".", 1)[-1]
        return metadata

    @staticmethod
    def _document_reference(document: Dict[str, Any]) -> Dict[str, Any]:
        reference = {
            key: document.get(key)
            for key in ("chunk_id", "source", "page", "backend")
            if document.get(key) is not None
        }
        score = document.get("score")
        if score is not None:
            try:
                reference["score"] = float(score)
            except (TypeError, ValueError):
                pass
        return reference


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