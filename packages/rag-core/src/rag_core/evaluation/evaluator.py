from dataclasses import dataclass
import json
from typing import List, Dict, Any, Optional

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from datasets import Dataset
    HAS_RAGAS = True
except ImportError:
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
        }


class RAGEvaluator:
    def __init__(self, llm_client: Any = None):
        if not HAS_RAGAS:
            raise ImportError(
                "RAGAS is not installed. Install it with: pip install ragas"
            )
        self.llm_client = llm_client

    def evaluate_response(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
        id: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate a single RAG response.

        Args:
            question: The input question.
            answer: The generated answer.
            contexts: List of retrieved context documents.
            ground_truth: The reference answer from validation data.
            id: Optional row identifier from the dataset.
            topic: Optional topic label from the dataset.

        Returns:
            EvaluationResult with metrics.
        """
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],  # RAGAS expects a list of lists
            "ground_truth": [ground_truth],
        }

        dataset = Dataset.from_dict(data)

        try:
            result = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ],
            )

            return EvaluationResult(
                id=id,
                topic=topic,
                question=question,
                answer=answer,
                ground_truth=ground_truth,
                contexts=contexts,
                faithfulness=result["faithfulness"][0],
                answer_relevancy=result["answer_relevancy"][0],
                context_precision=result["context_precision"][0],
                context_recall=result["context_recall"][0],
            )

        except Exception as e:
            print(f"Error evaluating response: {e}")
            return EvaluationResult(
                id=id,
                topic=topic,
                question=question,
                answer=answer,
                ground_truth=ground_truth,
                contexts=contexts,
            )

    def evaluate_batch(
        self,
        qa_pairs: List[Dict[str, Any]],
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple Q&A pairs.

        Args:
            qa_pairs: List of dicts with keys: question, answer, contexts, ground_truth,
                and optionally id/topic.

        Returns:
            List of EvaluationResult objects.
        """
        results = []
        for qa in qa_pairs:
            result = self.evaluate_response(
                question=qa["question"],
                answer=qa["answer"],
                contexts=qa.get("contexts", []),
                ground_truth=qa["ground_truth"],
                id=qa.get("id"),
                topic=qa.get("topic"),
            )
            results.append(result)
        return results

    def print_results(self, results: List[EvaluationResult]) -> None:
        """Print evaluation results in a human-readable format."""
        print("\n" + "="*80)
        print("RAGAS Evaluation Results")
        print("="*80)
        
        for i, result in enumerate(results, 1):
            print(f"\n[Question {i}]")
            print(f"Q: {result.question}")
            print(f"A: {result.answer}")
            print(f"\nMetrics:")
            print(f"  - Faithfulness:       {result.faithfulness:.4f}" if result.faithfulness else "  - Faithfulness:       N/A")
            print(f"  - Answer Relevancy:   {result.answer_relevancy:.4f}" if result.answer_relevancy else "  - Answer Relevancy:   N/A")
            print(f"  - Context Precision:  {result.context_precision:.4f}" if result.context_precision else "  - Context Precision:  N/A")
            print(f"  - Context Recall:     {result.context_recall:.4f}" if result.context_recall else "  - Context Recall:     N/A")
            
        print("\n" + "="*80 + "\n")

    def export_results(self, results: List[EvaluationResult], output_path: str) -> None:
        """Export results to JSON."""
        data = [r.to_dict() for r in results]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Results exported to {output_path}")
