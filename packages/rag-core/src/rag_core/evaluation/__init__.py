"""RAG evaluation utilities.

The deterministic retrieval-match helpers in `.matching` are import-cheap
(no LLM/RAGAS dependency) and are re-exported eagerly. The RAGAS-backed
`RAGEvaluator` requires `ragas` + `langchain-openai` + `datasets`; it is
exposed lazily via `__getattr__` so that scripts that only need the
retrieval helpers don't pay the import cost or fail if those deps are
missing.
"""
from .matching import (
    aggregate_metrics,
    candidate_matches_expected,
    compute_retrieval_metrics,
    expected_targets,
    normalize_optional_int,
)

__all__ = [
    "aggregate_metrics",
    "candidate_matches_expected",
    "compute_retrieval_metrics",
    "expected_targets",
    "normalize_optional_int",
    "RAGEvaluator",
    "EvaluationResult",
]


def __getattr__(name):
    """Lazy-import the RAGAS-backed evaluator to avoid pulling in heavy deps."""
    if name in {"RAGEvaluator", "EvaluationResult"}:
        from .evaluator import EvaluationResult, RAGEvaluator

        return {"RAGEvaluator": RAGEvaluator, "EvaluationResult": EvaluationResult}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")