#!/usr/bin/env python
"""
Example script: Evaluate RAG responses using RAGAS.
"""

import os
import sys
from typing import List, Dict, Any

from dotenv import load_dotenv

load_dotenv()

# Configure paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import pandas as pd
from rag_core.llm.transformers_client import TransformersClient
from rag_core.vectorstore.faiss_store import FaissStore
from rag_core.retrieval.hybrid_retriever import HybridRetriever
from rag_core.prompt_engineering.templates import load_templates
from rag_core.evaluation import RAGEvaluator


def load_evaluation_dataset(csv_path: str) -> List[Dict[str, Any]]:
    """Load evaluation examples from the main Golden Dataset CSV.

    The CSV must contain the columns: id, topic, question, ground_truth.
    Optional retrieval columns are: expected_sources, expected_pages,
    expected_chunk_ids, expected_slide, and related review fields.
    """
    df = pd.read_csv(
        csv_path,
        dtype={
            "id": str,
            "topic": str,
            "question": str,
            "ground_truth": str,
            "expected_sources": str,
            "expected_pages": str,
            "expected_chunk_ids": str,
            "expected_slide": str,
            "review_needed": str,
        },
    )

    required_columns = {"id", "topic", "question", "ground_truth"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Evaluation CSV is missing required columns: {missing_columns}")

    return df.to_dict(orient="records")


def generate_rag_responses(
    evaluation_data: List[Dict[str, Any]],
    faiss_index_path: str,
    faiss_meta_path: str,
    whoosh_index_dir: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Generate RAG answers for each evaluation example."""
    model = TransformersClient()
    faiss = FaissStore(
        dim=1536,
        index_path=faiss_index_path,
        meta_path=faiss_meta_path,
    )
    retriever = HybridRetriever(
        model_client=model,
        faiss_store=faiss,
        whoosh_index_dir=whoosh_index_dir,
    )
    templates = load_templates()

    qa_pairs: List[Dict[str, Any]] = []

    for example in evaluation_data:
        question = example["question"]
        documents = retriever.hybrid(question, k=top_k)
        contexts = [doc.get("text", "") for doc in documents]

        context_text = "\n\n".join(
            f"Source: {doc.get('source', 'unknown')}\nText: {doc.get('text', '')}"
            for doc in documents
        )

        prompt = templates["qa_prompt"].format(
            context=context_text,
            question=question,
        )
        answer = model.generate(prompt).strip()

        qa_pairs.append({
            "id": example.get("id"),
            "topic": example.get("topic"),
            "question": question,
            "ground_truth": example.get("ground_truth"),
            "answer": answer,
            "contexts": contexts,
            "retrieved_documents": documents,
            "top_k": top_k,
            "expected_sources": example.get("expected_sources"),
            "expected_pages": example.get("expected_pages"),
            "expected_chunk_ids": example.get("expected_chunk_ids"),
            "expected_slide": example.get("expected_slide"),
            "review_needed": example.get("review_needed"),
            "notes": example.get("notes"),
        })

        print(f"Generated response for ID={example.get('id')} question={question}")

    return qa_pairs


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RAG using RAGAS")
    parser.add_argument(
        "--evaluation-csv",
        required=True,
        help="Path to evaluation CSV containing id, topic, question, ground_truth",
    )
    parser.add_argument(
        "--faiss-index",
        default="data/index/faiss.index",
        help="Path to FAISS index",
    )
    parser.add_argument(
        "--faiss-meta",
        default="data/index/faiss_meta.pkl",
        help="Path to FAISS metadata",
    )
    parser.add_argument(
        "--whoosh-dir",
        default="data/index/whoosh",
        help="Path to Whoosh index",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved context documents",
    )
    parser.add_argument(
        "--output",
        default="data/evaluation_results.json",
        help="Output file for evaluation results",
    )
    
    args = parser.parse_args()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)
    
    evaluation_data = load_evaluation_dataset(args.evaluation_csv)
    
    print("Generating RAG responses...")
    qa_pairs = generate_rag_responses(
        evaluation_data=evaluation_data,
        faiss_index_path=args.faiss_index,
        faiss_meta_path=args.faiss_meta,
        whoosh_index_dir=args.whoosh_dir,
        top_k=args.top_k,
    )
    
    print("\nEvaluating responses with RAGAS...")
    evaluator = RAGEvaluator()
    results = evaluator.evaluate_batch(qa_pairs)
    
    evaluator.print_results(results)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    evaluator.export_results(results, args.output)


if __name__ == "__main__":
    main()
