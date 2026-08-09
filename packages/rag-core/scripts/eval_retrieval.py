#!/usr/bin/env python
"""Retrieval-only evaluation: precision@k + recall@k vs expected (source, page).

Runs hybrid retrieval (FAISS dense + Whoosh fuzzy, fused via RRF) over the
questions in the evaluation CSV and scores the retrieved chunks against the
expected source/page columns. No LLM generation, no RAGAS — deterministic.

Usage:
    python packages/rag-core/scripts/eval_retrieval.py \
        --evaluation-csv packages/rag-core/src/rag_core/evaluation/dataset.csv \
        --faiss-index data/index/faiss.index \
        --faiss-meta data/index/faiss_meta.pkl \
        --whoosh-dir data/index/whoosh \
        --top-k 5
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

# Path setup so we can import rag_core from this script's directory.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "src"))

import pandas as pd  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from rag_core.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from rag_core.llm.transformers_client import TransformersClient  # noqa: E402
from rag_core.vectorstore.faiss_store import FaissStore  # noqa: E402
from rag_core.evaluation.matching import (  # noqa: E402
    aggregate_metrics,
    compute_retrieval_metrics,
)


def load_evaluation_dataset(csv_path: str) -> List[Dict[str, Any]]:
    """Load evaluation examples from the golden CSV.

    Required columns: id, topic, question.
    Used by the matcher when present: expected_sources, expected_pages,
    expected_chunk_ids, expected_slide. ground_truth is loaded but ignored
    by the retrieval-only metrics.
    """
    df = pd.read_csv(csv_path)

    required = {"id", "topic", "question"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Evaluation CSV is missing required columns: {sorted(missing)}"
        )

    # Keep only the columns we care about.
    keep_cols = [
        c for c in df.columns
        if c in {
            "id", "topic", "question", "ground_truth",
            "expected_sources", "expected_pages",
            "expected_chunk_ids", "expected_slide",
            "review_needed", "notes",
        }
    ]
    return df[keep_cols].to_dict(orient="records")


def run_retrieval(
    examples: List[Dict[str, Any]],
    faiss_index_path: str,
    faiss_meta_path: str,
    whoosh_index_dir: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Run hybrid retrieval for each example; return QA records with retrieved docs."""
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

    qa_pairs: List[Dict[str, Any]] = []
    for example in examples:
        question = example["question"]
        documents = retriever.hybrid(question, k=top_k)

        qa_pairs.append({
            "id": example.get("id"),
            "topic": example.get("topic"),
            "question": question,
            "retrieved_documents": documents,
            "top_k": top_k,
            # Forward expected-* fields so the matcher can score this row.
            "expected_sources": example.get("expected_sources"),
            "expected_pages": example.get("expected_pages"),
            "expected_chunk_ids": example.get("expected_chunk_ids"),
            "expected_slide": example.get("expected_slide"),
        })

        first = documents[0] if documents else {}
        print(
            f"[id={example.get('id')}] retrieved {len(documents)} chunks | "
            f"top1={first.get('source', '?')}:p{first.get('page', '?')}"
        )

    return qa_pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieval-only evaluation: precision@k + recall@k"
    )
    parser.add_argument(
        "--evaluation-csv",
        required=True,
        help="CSV with id, topic, question, expected_sources, expected_pages",
    )
    parser.add_argument(
        "--faiss-index", default="data/index/faiss.index",
        help="Path to FAISS index",
    )
    parser.add_argument(
        "--faiss-meta", default="data/index/faiss_meta.pkl",
        help="Path to FAISS metadata pickle",
    )
    parser.add_argument(
        "--whoosh-dir", default="data/index/whoosh",
        help="Path to Whoosh index dir",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of chunks to retrieve per question",
    )
    parser.add_argument(
        "--output", default="data/retrieval_eval_results.json",
        help="Where to write the JSON results",
    )
    args = parser.parse_args()

    examples = load_evaluation_dataset(args.evaluation_csv)

    # Drop rows without expected_sources — we can't score them.
    scorable = [ex for ex in examples if ex.get("expected_sources")]
    skipped = len(examples) - len(scorable)
    if skipped:
        print(
            f"Skipping {skipped} rows without expected_sources "
            f"(no ground truth to score against)."
        )

    print(
        f"Running retrieval over {len(scorable)} questions "
        f"(top_k={args.top_k})..."
    )
    qa_pairs = run_retrieval(
        scorable,
        faiss_index_path=args.faiss_index,
        faiss_meta_path=args.faiss_meta,
        whoosh_index_dir=args.whoosh_dir,
        top_k=args.top_k,
    )

    per_row: List[Dict[str, Any]] = []
    for qa in qa_pairs:
        precision, recall, k = compute_retrieval_metrics(qa)
        per_row.append({
            "id": qa.get("id"),
            "topic": qa.get("topic"),
            "question": qa.get("question"),
            "retrieval_precision_at_k": precision,
            "retrieval_recall_at_k": recall,
            "retrieval_k": k,
            "expected_sources": qa.get("expected_sources"),
            "expected_pages": qa.get("expected_pages"),
            "expected_chunk_ids": qa.get("expected_chunk_ids"),
            "expected_slide": qa.get("expected_slide"),
            "retrieved_sources_pages": [
                {
                    "source": d.get("source"),
                    "page": d.get("page"),
                    "score": d.get("score"),
                }
                for d in (qa.get("retrieved_documents") or [])
            ],
        })

    summary = aggregate_metrics(per_row)

    print("\n" + "=" * 72)
    print("Retrieval-only evaluation")
    print("=" * 72)
    print(f"Rows evaluated:  {summary['rows_evaluated']}")
    if summary["mean_precision_at_k"] is not None:
        print(f"Mean precision@k: {summary['mean_precision_at_k']:.4f}")
        print(f"Mean recall@k:    {summary['mean_recall_at_k']:.4f}")
    print("=" * 72 + "\n")

    payload = {
        "summary": summary,
        "config": {
            "top_k": args.top_k,
            "evaluation_csv": args.evaluation_csv,
            "faiss_index": args.faiss_index,
            "faiss_meta": args.faiss_meta,
            "whoosh_dir": args.whoosh_dir,
        },
        "rows": per_row,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()