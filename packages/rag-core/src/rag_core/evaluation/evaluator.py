from typing import List, Dict, Any
from dataclasses import dataclass
import json

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
    faithfulness: float = None
    answer_relevancy: float = None
    context_precision: float = None
    context_recall: float = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
        }


class RAGEvaluator:
    def __init__(self, llm_client=None):
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
    ) -> EvaluationResult:
        """
        Evaluate a single RAG response.
        
        Args:
            question: The input question
            answer: The generated answer
            contexts: List of retrieved context documents
            
        Returns:
            EvaluationResult with metrics
        """
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [[contexts]],  # RAGAS expects nested lists
            "ground_truth": [answer],  # Use answer as ground truth for demo
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
            
            eval_result = EvaluationResult(
                question=question,
                answer=answer,
                contexts=contexts,
                faithfulness=result["faithfulness"][0],
                answer_relevancy=result["answer_relevancy"][0],
                context_precision=result["context_precision"][0],
                context_recall=result["context_recall"][0],
            )
            return eval_result
            
        except Exception as e:
            print(f"Error evaluating response: {e}")
            return EvaluationResult(
                question=question,
                answer=answer,
                contexts=contexts,
            )

    def evaluate_batch(
        self,
        qa_pairs: List[Dict[str, Any]],
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple Q&A pairs.
        
        Args:
            qa_pairs: List of dicts with keys: question, answer, contexts
            
        Returns:
            List of EvaluationResult objects
        """
        results = []
        for qa in qa_pairs:
            result = self.evaluate_response(
                question=qa["question"],
                answer=qa["answer"],
                contexts=qa.get("contexts", []),
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
