from rag_core.llm.transformers_client import TransformersClient
from rag_core.vectorstore.faiss_store import FaissStore
from rag_core.retrieval.hybrid_retriever import HybridRetriever
from rag_core.agents.agent_runner import AgentRunner
from rag_core.prompt_engineering.templates import load_templates

class RAGService:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = TransformersClient()

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

    def answer(self, question: str) -> str:
        result = self.answer_with_sources(question)
        return result["answer"]

    def answer_with_sources(self, question: str):
        docs = self.retriever.hybrid(question, k=5)

        if not docs:
            return {
                "answer": "I could not find the answer in the provided context.",
                "sources": [],
            }

        context = "\n\n".join(
            f"Source: {doc.get('source', 'unknown')}\nText: {doc.get('text', '')}"
            for doc in docs
        )

        prompt = self.templates["qa_prompt"].format(
            context=context,
            question=question
        )

        answer = self.model.generate(prompt)
        return {
            "answer": answer.strip() or "I could not find the answer in the provided context.",
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
