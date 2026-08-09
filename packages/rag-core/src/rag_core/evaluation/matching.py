"""Deterministic retrieval-match helpers.

These functions power both the RAGAS evaluator (precision@k / recall@k) and the
retrieval-only evaluation script. They have no LLM or RAGAS dependency, so they
can be imported standalone.
"""
from typing import Any, Dict, List, Optional, Tuple


def normalize_optional_int(value: Any) -> Optional[int]:
    """Return `value` as an int, or None if it can't be coerced."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _split_csv(value: Any, sep: str = ",") -> List[str]:
    """Split a delimited string into a list of trimmed non-empty tokens."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [token.strip() for token in value.split(sep) if token.strip()]
    return [str(value).strip()]


def expected_targets(qa: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build the list of expected targets for a single QA row.

    A target is either:
      - {"chunk_id": <id>}   — exact chunk match
      - {"source": <name>, "page"?: <int>, "slide"?: <int>}
                               — match by document filename + optional page/slide

    Sources of inputs (from the CSV):
      - expected_chunk_ids (comma-separated)
      - expected_sources (comma-separated)
      - expected_pages (semicolon-separated list; only the first is used today)
      - expected_slide (single int)
    """
    targets: List[Dict[str, Any]] = []

    for chunk_id in _split_csv(qa.get("expected_chunk_ids")):
        targets.append({"chunk_id": chunk_id})

    expected_sources = _split_csv(qa.get("expected_sources"))
    if expected_sources:
        # `expected_pages` may list multiple candidate pages (semicolon-separated)
        # when a concept spans a page range, e.g. "15;16". We emit one target per
        # (source, page) pair so any of them can satisfy the matcher.
        page_values = _split_csv(qa.get("expected_pages"), sep=";")
        expected_pages_list = [
            normalize_optional_int(v) for v in page_values
        ] if page_values else [None]
        expected_slide = normalize_optional_int(qa.get("expected_slide"))

        for source in expected_sources:
            for expected_page in expected_pages_list:
                target: Dict[str, Any] = {"source": source}
                if expected_page is not None:
                    target["page"] = expected_page
                if expected_slide is not None:
                    target["slide"] = expected_slide
                targets.append(target)

    return targets


def candidate_matches_expected(
    document: Dict[str, Any],
    expected_targets: List[Dict[str, Any]],
    page_tolerance: int = 0,
) -> bool:
    """True iff `document` satisfies any of the expected targets.

    `page_tolerance` widens the page match: a doc page p is considered to match
    expected page e when |p - e| <= page_tolerance. Used for "concept is
    discussed nearby" style metrics. Tolerance does NOT affect chunk_id
    matches (those are exact by design).
    """
    if not expected_targets:
        return False

    for target in expected_targets:
        if "chunk_id" in target:
            doc_chunk_id = document.get("chunk_id") or document.get("id")
            if (
                doc_chunk_id is not None
                and str(doc_chunk_id) == str(target["chunk_id"])
            ):
                return True
            continue

        if "source" in target:
            if str(document.get("source", "")).strip() != str(target["source"]).strip():
                continue
            if "page" in target:
                doc_page = normalize_optional_int(document.get("page"))
                expected_page = target["page"]
                if doc_page is None or expected_page is None:
                    if doc_page != expected_page:
                        continue
                elif abs(doc_page - expected_page) > page_tolerance:
                    continue
            if "slide" in target:
                doc_slide = normalize_optional_int(document.get("slide"))
                if doc_slide != target["slide"]:
                    continue
            return True

    return False


def compute_retrieval_metrics(
    qa: Dict[str, Any],
    page_tolerance: int = 0,
) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    """Compute precision@k and recall@k for a single QA row.

    `page_tolerance` widens the page match by ±N pages (default 0 = exact).
    Returns (precision_at_k, recall_at_k, k) or (None, None, None) when no
    retrieved documents / expected targets are available.
    """
    retrieved_documents = qa.get("retrieved_documents") or []
    if not retrieved_documents:
        return None, None, None

    targets = expected_targets(qa)
    if not targets:
        return None, None, None

    k = normalize_optional_int(qa.get("top_k"))
    if k is None:
        k = len(retrieved_documents)
    if k <= 0:
        k = 1
    k = min(k, len(retrieved_documents))

    relevant_hits = sum(
        1 for doc in retrieved_documents[:k]
        if candidate_matches_expected(doc, targets, page_tolerance=page_tolerance)
    )

    precision_at_k = relevant_hits / float(k)
    recall_at_k = relevant_hits / float(len(targets))
    return precision_at_k, recall_at_k, k


def aggregate_metrics(
    per_row: List[Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    """Aggregate precision@k / recall@k across rows (ignoring Nones)."""
    precisions = [
        row["retrieval_precision_at_k"]
        for row in per_row
        if row.get("retrieval_precision_at_k") is not None
    ]
    recalls = [
        row["retrieval_recall_at_k"]
        for row in per_row
        if row.get("retrieval_recall_at_k") is not None
    ]
    return {
        "rows_evaluated": len(precisions),
        "mean_precision_at_k": (sum(precisions) / len(precisions)) if precisions else None,
        "mean_recall_at_k": (sum(recalls) / len(recalls)) if recalls else None,
    }