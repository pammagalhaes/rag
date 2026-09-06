import os
from dataclasses import dataclass
import json
from typing import List, Dict, Any, Optional, Tuple
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import traceback

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
    # Allow this module to be imported even when RAGAS is not installed.
    traceback.print_exc()
    HAS_RAGAS = False


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

    def _normalize_optional_int(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, float) and (value != value):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _normalize_text_list(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            parts = [item.strip() for item in value.replace(";", ",").split(",") if item and item.strip()]
            return parts
        if isinstance(value, list):
            return [str(item).strip() for item in value if item is not None and str(item).strip()]
        if value != value:
            return []
        text = str(value).strip()
        return [text] if text else []

    def _expected_targets(self, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        targets: List[Dict[str, Any]] = []

        expected_chunk_ids = self._normalize_text_list(qa.get("expected_chunk_ids"))
        for chunk_id in expected_chunk_ids:
            targets.append({"chunk_id": chunk_id})

        expected_sources = self._normalize_text_list(qa.get("expected_sources"))
        expected_pages = qa.get("expected_pages")
        page_values: List[int] = []

        if expected_pages is None:
            page_values = []
        elif isinstance(expected_pages, str):
            page_values = [
                self._normalize_optional_int(value)
                for value in expected_pages.split(";")
                if value and value.strip()
            ]
        elif isinstance(expected_pages, list):
            page_values = [self._normalize_optional_int(value) for value in expected_pages]
        else:
            page_values = [self._normalize_optional_int(expected_pages)]

        page_values = [page for page in page_values if page is not None]

        if not expected_sources and not page_values:
            return targets

        if not expected_sources:
            expected_sources = [""]

        if not page_values:
            for expected_source in expected_sources:
                target = {}
                if expected_source:
                    target["source"] = expected_source
                targets.append(target)
            return targets

        for expected_source in expected_sources:
            for expected_page in page_values:
                target: Dict[str, Any] = {}
                if expected_source:
                    target["source"] = expected_source
                target["page"] = expected_page

                expected_slide = self._normalize_optional_int(qa.get("expected_slide"))
                if expected_slide is not None:
                    target["slide"] = expected_slide
                targets.append(target)

        return targets

    def _candidate_matches_expected(self, document: Dict[str, Any], expected_targets: List[Dict[str, Any]]) -> bool:
        if not expected_targets:
            return False

        for target in expected_targets:
            if "chunk_id" in target:
                doc_chunk_id = document.get("chunk_id") or document.get("id")
                if doc_chunk_id is not None and str(doc_chunk_id) == str(target["chunk_id"]):
                    return True

            if "source" in target:
                doc_source = str(document.get("source", "")).strip()
                expected_source = str(target.get("source", "")).strip()
                if expected_source and doc_source != expected_source:
                    continue
                if "page" in target:
                    doc_page = self._normalize_optional_int(document.get("page"))
                    if doc_page != target["page"]:
                        continue
                if "slide" in target:
                    doc_slide = self._normalize_optional_int(document.get("slide"))
                    if doc_slide != target["slide"]:
                        continue
                return True

        return False

    def _compute_retrieval_metrics(self, qa: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[int]]:
        retrieved_documents = qa.get("retrieved_documents") or []
        if not retrieved_documents:
            return None, None, None

        expected_targets = self._expected_targets(qa)
        if not expected_targets:
            return None, None, None

        # retrieval-only evaluation should not print debug logs by default

        k = self._normalize_optional_int(qa.get("top_k"))
        if k is None:
            k = len(retrieved_documents)
        if k <= 0:
            k = 1
        k = min(k, len(retrieved_documents))

        relevant_hits = 0
        for index, doc in enumerate(retrieved_documents[:k]):
            match_result = self._candidate_matches_expected(doc, expected_targets)

            if match_result:
                relevant_hits += 1
                continue

            doc_source = str(doc.get("source", "")).strip()
            doc_page = self._normalize_optional_int(doc.get("page"))
            doc_slide = self._normalize_optional_int(doc.get("slide"))
            doc_chunk_id = doc.get("chunk_id") or doc.get("id")

            for target in expected_targets:
                target_source = str(target.get("source", "")).strip()
                target_page = target.get("page")
                target_slide = target.get("slide")
                target_chunk_id = target.get("chunk_id")

                reason_parts = []
                if target_chunk_id is not None:
                    if doc_chunk_id is None or str(doc_chunk_id) != str(target_chunk_id):
                        reason_parts.append("chunk_id")
                if target_source:
                    if doc_source != target_source:
                        reason_parts.append("source")
                if target_page is not None:
                    if doc_page != target_page:
                        reason_parts.append("page")
                if target_slide is not None:
                    if doc_slide != target_slide:
                        reason_parts.append("slide")

                # do not print debug information in production evaluator
                # (reason_parts can be inspected by callers if needed)
                if reason_parts:
                    pass

        precision_at_k = relevant_hits / float(k)
        recall_at_k = relevant_hits / float(len(expected_targets))
        return precision_at_k, recall_at_k, k

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
                precision_at_k, recall_at_k, retrieval_k = self._compute_retrieval_metrics(qa)
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


class RetrievalEvaluator:
    """Standalone retrieval-only evaluator that computes Precision@K and Recall@K
    using `expected_chunk_ids` as primary ground truth. This class does not
    depend on RAGAS and can be used for fast retrieval-only evaluation.
    """

    @staticmethod
    def _normalize_optional_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, float) and (value != value):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_text_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            parts = [item.strip() for item in value.split(",") if item and item.strip()]
            return parts
        if isinstance(value, list):
            return [str(item).strip() for item in value if item is not None and str(item).strip()]
        if value != value:
            return []
        text = str(value).strip()
        return [text] if text else []

    @classmethod
    def compute_metrics_for_example(cls, qa: Dict[str, Any], retrieved_documents: List[Dict[str, Any]], ks: List[int]) -> Dict[int, Dict[str, Any]]:
        """Compute precision/recall for multiple Ks for a single QA example.

        - If `expected_chunk_ids` are present, they are used as the sole ground truth.
        - Otherwise falls back to page/source matching for relevance.
        Returns a mapping K -> {precision, recall, relevant_hits, expected_count, retrieved_ids}
        """
        expected_chunk_ids = cls._normalize_text_list(qa.get("expected_chunk_ids"))
        max_k = max(ks)
        docs = retrieved_documents[:max_k]

        # helper to extract chunk id from a retrieved document
        def doc_chunk_id(d: Dict[str, Any]) -> Optional[str]:
            return d.get("chunk_id") or d.get("id")

        results: Dict[int, Dict[str, Any]] = {}

        if expected_chunk_ids:
            expected_set = set([str(x) for x in expected_chunk_ids])
            for k in ks:
                topk = docs[:k]
                k_used = min(k, len(topk))
                retrieved_ids = [doc_chunk_id(d) for d in topk]
                relevant_hits = sum(1 for rid in retrieved_ids if rid is not None and str(rid) in expected_set)
                precision = relevant_hits / float(k_used) if k_used > 0 else 0.0
                recall = relevant_hits / float(len(expected_set)) if expected_set else 0.0
                results[k] = {
                    "precision": precision,
                    "recall": recall,
                    "relevant_hits": relevant_hits,
                    "expected_count": len(expected_set),
                    "retrieved_ids": retrieved_ids,
                }
            return results

        # Fallback: use expected_pages/sources matching similar to older behavior
        expected_sources = cls._normalize_text_list(qa.get("expected_sources"))
        expected_pages_raw = qa.get("expected_pages")
        page_values: List[int] = []
        if expected_pages_raw is None:
            page_values = []
        elif isinstance(expected_pages_raw, str):
            page_values = [cls._normalize_optional_int(v) for v in expected_pages_raw.split(";") if v and v.strip()]
        elif isinstance(expected_pages_raw, list):
            page_values = [cls._normalize_optional_int(v) for v in expected_pages_raw]
        else:
            page_values = [cls._normalize_optional_int(expected_pages_raw)]
        page_values = [p for p in page_values if p is not None]

        # build expected target set of (source,page) when available
        expected_targets = []
        if not expected_sources and not page_values:
            expected_targets = []
        else:
            if not expected_sources:
                expected_sources = [""]
            if not page_values:
                for src in expected_sources:
                    expected_targets.append({"source": src})
            else:
                for src in expected_sources:
                    for p in page_values:
                        expected_targets.append({"source": src, "page": p})

        for k in ks:
            topk = docs[:k]
            k_used = min(k, len(topk))
            retrieved_ids = [doc_chunk_id(d) for d in topk]
            relevant_hits = 0
            for d in topk:
                # match by source/page
                matched = False
                for t in expected_targets:
                    if "source" in t:
                        doc_source = str(d.get("source", "")).strip()
                        target_source = str(t.get("source", "")).strip()
                        if target_source and doc_source != target_source:
                            continue
                    if "page" in t:
                        doc_page = cls._normalize_optional_int(d.get("page"))
                        if doc_page != t.get("page"):
                            continue
                    matched = True
                    break
                if matched:
                    relevant_hits += 1

            precision = relevant_hits / float(k_used) if k_used > 0 else 0.0
            recall = relevant_hits / float(len(expected_targets)) if expected_targets else 0.0
            results[k] = {
                "precision": precision,
                "recall": recall,
                "relevant_hits": relevant_hits,
                "expected_count": len(expected_targets),
                "retrieved_ids": retrieved_ids,
            }

        return results

    @classmethod
    def evaluate_dataset(cls, evaluation_data: List[Dict[str, Any]], retriever: Any, ks: List[int]) -> Tuple[Dict[int, Dict[str, float]], List[Dict[str, Any]]]:
        """Evaluate an entire dataset using the provided retriever.

        Returns (aggregate_metrics_by_k, per_question_details)
        """
        max_k = max(ks)
        per_question = []
        # accumulators for averaging
        agg = {k: {"precision_sum": 0.0, "recall_sum": 0.0, "count": 0} for k in ks}

        for example in evaluation_data:
            question = example.get("question", "")
            docs = retriever.hybrid(question, k=max_k)
            metrics = cls.compute_metrics_for_example(example, docs, ks)

            detail = {
                "id": example.get("id"),
                "question": question,
                "expected_chunk_ids": cls._normalize_text_list(example.get("expected_chunk_ids")),
                "retrieved_chunk_ids": metrics[ks[0]]["retrieved_ids"] if ks else [],
                "metrics": metrics,
            }
            per_question.append(detail)

            for k in ks:
                agg[k]["precision_sum"] += metrics[k]["precision"]
                agg[k]["recall_sum"] += metrics[k]["recall"]
                agg[k]["count"] += 1

        aggregate = {}
        for k in ks:
            count = agg[k]["count"]
            if count:
                aggregate[k] = {
                    "precision": agg[k]["precision_sum"] / count,
                    "recall": agg[k]["recall_sum"] / count,
                }
            else:
                aggregate[k] = {"precision": 0.0, "recall": 0.0}

        return aggregate, per_question
