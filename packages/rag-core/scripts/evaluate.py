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
from rag_core.evaluation import RetrievalEvaluator


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
        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")

        for i, doc in enumerate(documents, 1):
            print(f"\nDOCUMENT {i}")
            print(doc)

        print("=" * 80)
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


def run_retrieval_only(
    evaluation_data: List[Dict[str, Any]],
    faiss_index_path: str,
    faiss_meta_path: str,
    whoosh_index_dir: str,
    ks: List[int],
):
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

    aggregate, per_question = RetrievalEvaluator.evaluate_dataset(evaluation_data, retriever, ks)

    print("\n" + "=" * 80)
    print("Retrieval Evaluation")
    print("=" * 80)
    for k in ks:
        metrics = aggregate.get(k, {})
        print(f"K={k} Precision={metrics.get('precision'):.4f} Recall={metrics.get('recall'):.4f}")

    print("\nPer-question details:")
    for detail in per_question:
        print("-" * 60)
        print(f"ID: {detail.get('id')}\nQ: {detail.get('question')}")
        print(f"Expected chunk IDs: {detail.get('expected_chunk_ids')}")
        for k in ks:
            m = detail['metrics'][k]
            print(f"  K={k} -> relevant={m['relevant_hits']} expected={m['expected_count']} precision={m['precision']:.4f} recall={m['recall']:.4f}")


def find_chunks(query: str, faiss_index_path: str, faiss_meta_path: str, whoosh_index_dir: str, k: int = 20):
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

    docs = retriever.hybrid(query, k=k)
    print("Found candidates:\n")
    for i, doc in enumerate(docs, 1):
        cid = doc.get('chunk_id') or doc.get('id')
        src = doc.get('source')
        page = doc.get('page')
        text = str(doc.get('text', ''))[:150].replace('\n', ' ')
        print(f"RANK {i} | chunk_id={cid} | source={src} | page={page} | preview={text}")


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
        "--retrieval-only",
        action="store_true",
        help="Run retrieval-only evaluation (no RAGAS, no LLM calls)",
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        nargs='+',
        default=[5,10,15,20],
        help="List of K values for retrieval metrics (e.g. --retrieval-k 5 10 15 20)",
    )
    parser.add_argument(
        "--find-chunks",
        type=str,
        default=None,
        help="Find candidate chunks for the given query and print chunk_ids",
    )
    parser.add_argument(
        "--output",
        default="data/evaluation_results.json",
        help="Output file for evaluation results",
    )
    
    args = parser.parse_args()
    
    # Only require OpenAI key when running full RAGAS evaluation
    if not args.retrieval_only and not args.find_chunks:
        if not os.getenv("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY not set")
            sys.exit(1)
    
    evaluation_data = load_evaluation_dataset(args.evaluation_csv)
    
    if args.find_chunks:
        find_chunks(args.find_chunks, args.faiss_index, args.faiss_meta, args.whoosh_dir, k=args.top_k)
        return

    if args.retrieval_only:
        print("Running retrieval-only evaluation...")
        run_retrieval_only(
            evaluation_data=evaluation_data,
            faiss_index_path=args.faiss_index,
            faiss_meta_path=args.faiss_meta,
            whoosh_index_dir=args.whoosh_dir,
            ks=args.retrieval_k,
        )
        return

    # Full RAGAS evaluation path
    print("Generating RAG responses...")
    qa_pairs = generate_rag_responses(
        evaluation_data=evaluation_data,
        faiss_index_path=args.faiss_index,
        faiss_meta_path=args.faiss_meta,
        whoosh_index_dir=args.whoosh_dir,
        top_k=args.top_k,
    )

    print("\nEvaluating responses with RAGAS...")
    # import RAGEvaluator lazily to ensure ragas is available for RAGAS path
    from rag_core.evaluation import RAGEvaluator

    evaluator = RAGEvaluator()
    results = evaluator.evaluate_batch(qa_pairs)

    evaluator.print_results(results)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    evaluator.export_results(results, args.output)


if __name__ == "__main__":
    main()
