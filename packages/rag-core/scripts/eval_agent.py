#!/usr/bin/env python
"""Evaluate the optional LangGraph agent against the golden dataset."""

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "packages", "rag-core", "src"))
sys.path.insert(0, os.path.join(_ROOT, "apps", "backend", "src"))
sys.path.insert(0, os.path.join(_ROOT, "apps", "backend"))

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(_ROOT, ".env"))

from configs.loader import load_config
from rag_backend.services.rag_service import RAGService
from rag_core.evaluation.matching import compute_retrieval_metrics


def load_dataset(path: str) -> List[Dict[str, Any]]:
    frame = pd.read_csv(path)
    required = {"id", "topic", "question", "ground_truth"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Evaluation CSV is missing columns: {sorted(missing)}")
    return frame.to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the optional LangGraph agent")
    parser.add_argument("--evaluation-csv", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default="data/agent_eval_results.json")
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument(
        "--free-agent",
        action="store_true",
        help="Evaluate agent-selected retrieval without the baseline hybrid anchor",
    )
    args = parser.parse_args()

    cfg = load_config("default.yaml")
    cfg.setdefault("agent", {})["enabled"] = True
    cfg["agent"]["max_candidates"] = max(args.top_k, cfg["agent"].get("max_candidates", 10))
    cfg["agent"]["protect_baseline"] = not args.free_agent
    service = RAGService(cfg)
    examples = load_dataset(args.evaluation_csv)

    rows = []
    ragas_pairs = []
    for example in examples:
        started = time.perf_counter()
        result = service.langgraph_agent.run(example["question"], top_k=args.top_k)
        latency_ms = (time.perf_counter() - started) * 1000
        qa = {
            **example,
            "retrieved_documents": result.get("retrieved_documents", []),
            "top_k": args.top_k,
        }
        precision, recall, k = compute_retrieval_metrics(qa)
        rows.append({
            "id": example.get("id"),
            "topic": example.get("topic"),
            "question": example["question"],
            "answer": result.get("answer"),
            "retrieval_precision_at_k": precision,
            "retrieval_recall_at_k": recall,
            "retrieval_k": k,
            "latency_ms": latency_ms,
            "search_query": result.get("search_query"),
            "search_type": result.get("search_type"),
            "use_rerank": result.get("use_rerank", False),
            "expected_chunk_ids": example.get("expected_chunk_ids"),
            "retrieved_documents": result.get("retrieved_documents", []),
        })
        ragas_pairs.append({
            "id": example.get("id"),
            "topic": example.get("topic"),
            "question": example["question"],
            "answer": result.get("answer", ""),
            "contexts": [d.get("text", "") for d in result.get("retrieved_documents", [])],
            "ground_truth": example.get("ground_truth", ""),
            "retrieved_documents": result.get("retrieved_documents", []),
            "top_k": args.top_k,
            "expected_chunk_ids": example.get("expected_chunk_ids"),
        })

    valid_precision = [r["retrieval_precision_at_k"] for r in rows if r["retrieval_precision_at_k"] is not None]
    valid_recall = [r["retrieval_recall_at_k"] for r in rows if r["retrieval_recall_at_k"] is not None]
    payload: Dict[str, Any] = {
        "retrieval_summary": {
            "rows_evaluated": len(rows),
            "mean_precision_at_k": sum(valid_precision) / len(valid_precision) if valid_precision else None,
            "mean_recall_at_k": sum(valid_recall) / len(valid_recall) if valid_recall else None,
        },
        "config": {
            "top_k": args.top_k,
            "evaluation_csv": args.evaluation_csv,
            "agent": True,
            "protect_baseline": not args.free_agent,
        },
        "rows": rows,
    }

    if not args.skip_ragas:
        from rag_core.evaluation import RAGEvaluator

        ragas_results = RAGEvaluator().evaluate_batch(ragas_pairs)
        payload["ragas"] = [result.to_dict() for result in ragas_results]
        metric_names = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]
        payload["ragas_summary"] = {
            f"mean_{metric}": _mean_metric(ragas_results, metric)
            for metric in metric_names
        }
        payload["ragas_summary"]["valid_rows"] = {
            metric: _valid_metric_count(ragas_results, metric)
            for metric in metric_names
        }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, default=str)
    print(f"Agent evaluation written to {args.output}")


def _mean_metric(results: List[Any], metric_name: str):
    values = [
        float(getattr(result, metric_name))
        for result in results
        if getattr(result, metric_name) is not None
        and math.isfinite(float(getattr(result, metric_name)))
    ]
    return sum(values) / len(values) if values else None


def _valid_metric_count(results: List[Any], metric_name: str) -> int:
    return sum(
        1
        for result in results
        if getattr(result, metric_name) is not None
        and math.isfinite(float(getattr(result, metric_name)))
    )


if __name__ == "__main__":
    main()