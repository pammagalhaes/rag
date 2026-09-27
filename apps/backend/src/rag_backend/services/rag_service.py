from rag_core.llm.transformers_client import TransformersClient
from rag_core.vectorstore.faiss_store import FaissStore
from rag_core.retrieval.hybrid_retriever import HybridRetriever
from rag_core.agents.agent_runner import AgentRunner
from rag_core.observability.langfuse_tracer import LangfuseTracer
from rag_core.prompt_engineering.templates import load_templates

class RAGService:
    def __init__(self, cfg):
        self.cfg = cfg
        observability_cfg = cfg.get("observability", {})
        self.tracer = LangfuseTracer(
            enabled=observability_cfg.get("langfuse_enabled", False),
            capture_content=observability_cfg.get("capture_content", False),
        )
        self.model = TransformersClient(
            temperature=cfg.get("model", {}).get("temperature", 0.0),
            tracer=self.tracer,
        )

        self.faiss = FaissStore(
            dim=1536,
            index_path=cfg["vectorstore"]["faiss_index_path"],
            meta_path=cfg["vectorstore"]["faiss_meta_path"]
        )

        self.retriever = HybridRetriever(
            model_client=self.model,
            faiss_store=self.faiss,
            whoosh_index_dir="data/index/whoosh"
        )

        self.templates = load_templates()

        self.agent = AgentRunner(
            model_client=self.model,
            retriever=self.retriever,
            prompt_templates=self.templates
        )
        self.langgraph_agent = None
        agent_cfg = cfg.get("agent", {})
        if agent_cfg.get("enabled", False):
            from rag_core.agents.langgraph_agent import LangGraphAgent

            self.langgraph_agent = LangGraphAgent(
                model_client=self.model,
                retriever=self.retriever,
                prompt_templates=self.templates,
                max_candidates=agent_cfg.get("max_candidates", 10),
                protect_baseline=agent_cfg.get("protect_baseline", True),
                rerank_enabled=agent_cfg.get("rerank_enabled", True),
                tracer=self.tracer,
            )

    def answer(self, question: str) -> str:
        result = self.answer_with_sources(question)
        return result["answer"]

    def answer_with_sources(self, question: str):
        if self.langgraph_agent is not None:
            return self.langgraph_agent.run(question)

        with self.tracer.observation(
            "rag.request",
            input_data=question,
            metadata={"agent_enabled": False},
        ) as trace:
            with self.tracer.observation(
                "retrieval.hybrid",
                input_data=question,
                metadata={"top_k": 5},
            ) as retrieval_span:
                docs = self.retriever.hybrid(question, k=5)
                retrieval_span.update(
                    metadata={
                        "candidate_count": len(docs),
                        "chunk_ids": [doc.get("chunk_id") for doc in docs],
                    }
                )

            if not docs:
                answer = "I could not find the answer in the provided context."
                trace.update(metadata={"retrieved_count": 0})
                return {"answer": answer, "sources": []}

            context = "\n\n".join(
                f"Source: {doc.get('source', 'unknown')}\nText: {doc.get('text', '')}"
                for doc in docs
            )

            prompt = self.templates["qa_prompt"].format(
                context=context,
                question=question
            )

            answer = self.model.generate(prompt).strip()
            answer = answer or "I could not find the answer in the provided context."
            trace.update(
                metadata={
                    "retrieved_count": len(docs),
                    "chunk_ids": [doc.get("chunk_id") for doc in docs],
                },
                **({"output": answer} if self.tracer.capture_content else {}),
            )
            return {
                "answer": answer,
                "sources": [
                    {
                        "source": doc.get("source"),
                        "page": doc.get("page"),
                        "slide": doc.get("slide"),
                        "chunk_id": doc.get("chunk_id"),
                        "backend": doc.get("backend"),
                        "score": doc.get("score"),
                    }
                    for doc in docs
                ],
            }
