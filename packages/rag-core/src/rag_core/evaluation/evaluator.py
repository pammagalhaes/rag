import os
from dataclasses import dataclass
import json
from typing import List, Dict, Any, Optional, Tuple
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import traceback

from .matching import compute_retrieval_metrics

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from datasets import Dataset

    HAS_RAGAS = True

except Exception:
    traceback.print_exc()
    raise


@dataclass
class EvaluationResult:
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None
    id: Optional[str] = None
    topic: Optional[str] = None
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    retrieved_documents: Optional[List[Dict[str, Any]]] = None
    retrieval_precision_at_k: Optional[float] = None
    retrieval_recall_at_k: Optional[float] = None
    retrieval_k: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "answer": self.answer,
            "contexts": self.contexts,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "retrieved_documents": self.retrieved_documents,
            "retrieval_precision_at_k": self.retrieval_precision_at_k,
            "retrieval_recall_at_k": self.retrieval_recall_at_k,
            "retrieval_k": self.retrieval_k,
        }


class RAGEvaluator:
    def __init__(self, llm_client: Any = None):
        if not HAS_RAGAS:
            raise ImportError(
                "RAGAS is not installed. Install it with: pip install ragas"
            )

        self.llm_client = llm_client

        # RAGAS-compatible wrappers for v0.4.3
        self.ragas_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
            )
        )

        self.ragas_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model="text-embedding-3-small"
            )
        )

    def _build_ragas_runtime(self) -> Tuple[Any, Any]:
        """Create explicit LLM and embedding adapters compatible with RAGAS 0.4.3."""
        try:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "langchain-openai is required for RAGAS evaluation."
                " Install it with: pip install langchain-openai"
            ) from exc

        if self.llm_client is not None and hasattr(self.llm_client, "invoke"):
            llm = LangchainLLMWrapper(self.llm_client)
        else:
            llm = LangchainLLMWrapper(
                ChatOpenAI(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    temperature=0,
                )
            )

        embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            )
        )

        return llm, embeddings

    def evaluate_response(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
        id: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> EvaluationResult:
        """Evaluate a single RAG response using RAGAS 0.4.3-compatible fields."""
        return self.evaluate_batch(
            [
                {
                    "question": question,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": ground_truth,
                    "id": id,
                    "topic": topic,
                }
            ]
        )[0]

    def evaluate_batch(
        self,
        qa_pairs: List[Dict[str, Any]],
    ) -> List[EvaluationResult]:
        """Evaluate multiple Q&A pairs in a single RAGAS call."""
        if not qa_pairs:
            return []

        records = []
        for qa in qa_pairs:
            records.append(
                {
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "contexts": qa.get("contexts", []),
                    "retrieved_contexts": qa.get("contexts", []),
                    "ground_truth": qa["ground_truth"],
                }
            )

        dataset = Dataset.from_list(records)

        try:
            llm, embeddings = self._build_ragas_runtime()
            result = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ],
                llm=llm,
                embeddings=embeddings,
            )

            results = []
            for index, qa in enumerate(qa_pairs):
                precision_at_k, recall_at_k, retrieval_k = compute_retrieval_metrics(qa)
                results.append(
                    EvaluationResult(
                        id=qa.get("id"),
                        topic=qa.get("topic"),
                        question=qa["question"],
                        answer=qa["answer"],
                        ground_truth=qa["ground_truth"],
                        contexts=qa.get("contexts", []),
                        faithfulness=result["faithfulness"][index],
                        answer_relevancy=result["answer_relevancy"][index],
                        context_precision=result["context_precision"][index],
                        context_recall=result["context_recall"][index],
                        retrieved_documents=qa.get("retrieved_documents", []),
                        retrieval_precision_at_k=precision_at_k,
                        retrieval_recall_at_k=recall_at_k,
                        retrieval_k=retrieval_k,
                    )
                )
            return results

        except Exception as e:
            print(f"Error evaluating batch: {e}")
            return [
                EvaluationResult(
                    id=qa.get("id"),
                    topic=qa.get("topic"),
                    question=qa["question"],
                    answer=qa["answer"],
                    ground_truth=qa["ground_truth"],
                    contexts=qa.get("contexts", []),
                    retrieved_documents=qa.get("retrieved_documents", []),
                )
                for qa in qa_pairs
            ]

    def print_results(self, results: List[EvaluationResult]) -> None:
        """Print evaluation results in a human-readable format."""
        print("\n" + "=" * 80)
        print("RAGAS Evaluation Results")
        print("=" * 80)

        for i, result in enumerate(results, 1):
            print(f"\n[Question {i}]")
            print(f"Q: {result.question}")
            print(f"A: {result.answer}")
            print("\nMetrics:")
            print(
                f"  - Faithfulness:       {result.faithfulness:.4f}"
                if result.faithfulness is not None
                else "  - Faithfulness:       N/A"
            )
            print(
                f"  - Answer Relevancy:   {result.answer_relevancy:.4f}"
                if result.answer_relevancy is not None
                else "  - Answer Relevancy:   N/A"
            )
            print(
                f"  - Context Precision:  {result.context_precision:.4f}"
                if result.context_precision is not None
                else "  - Context Precision:  N/A"
            )
            print(
                f"  - Context Recall:     {result.context_recall:.4f}"
                if result.context_recall is not None
                else "  - Context Recall:     N/A"
            )
            print(
                f"  - Retrieval Precision@{result.retrieval_k}: {result.retrieval_precision_at_k:.4f}"
                if result.retrieval_precision_at_k is not None
                else "  - Retrieval Precision@K: N/A"
            )
            print(
                f"  - Retrieval Recall@{result.retrieval_k}:    {result.retrieval_recall_at_k:.4f}"
                if result.retrieval_recall_at_k is not None
                else "  - Retrieval Recall@K:    N/A"
            )

        print("\n" + "=" * 80 + "\n")

    def export_results(self, results: List[EvaluationResult], output_path: str) -> None:
        """Export results to JSON."""
        data = [r.to_dict() for r in results]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Results exported to {output_path}")
