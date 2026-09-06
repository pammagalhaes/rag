from rag_core.evaluation.evaluator import RetrievalEvaluator


def test_single_expected_chunk():
    qa = {"expected_chunk_ids": "chunk-a"}
    retrieved = [{"chunk_id": "chunk-a"}, {"chunk_id": "chunk-b"}]
    ks = [1, 2]
    metrics = RetrievalEvaluator.compute_metrics_for_example(qa, retrieved, ks)

    assert metrics[1]["precision"] == 1.0
    assert metrics[1]["recall"] == 1.0

    assert metrics[2]["precision"] == 0.5
    assert metrics[2]["recall"] == 1.0


def test_multiple_expected_chunks():
    qa = {"expected_chunk_ids": "a,b"}
    retrieved = [{"chunk_id": "b"}, {"chunk_id": "c"}, {"chunk_id": "a"}]
    ks = [2, 3]
    metrics = RetrievalEvaluator.compute_metrics_for_example(qa, retrieved, ks)

    # K=2: only 'b' found
    assert metrics[2]["relevant_hits"] == 1
    assert abs(metrics[2]["precision"] - 0.5) < 1e-6
    assert abs(metrics[2]["recall"] - 0.5) < 1e-6

    # K=3: 'b' and 'a' found
    assert metrics[3]["relevant_hits"] == 2
    assert abs(metrics[3]["precision"] - (2.0 / 3.0)) < 1e-6
    assert abs(metrics[3]["recall"] - 1.0) < 1e-6


def test_no_expected_chunk_fallback_to_pages():
    qa = {"expected_chunk_ids": None, "expected_sources": "book.pdf", "expected_pages": "5"}
    retrieved = [
        {"source": "book.pdf", "page": 5},
        {"source": "book.pdf", "page": 6},
    ]
    ks = [1, 2]
    metrics = RetrievalEvaluator.compute_metrics_for_example(qa, retrieved, ks)

    assert metrics[1]["relevant_hits"] == 1
    assert abs(metrics[1]["precision"] - 1.0) < 1e-6
    assert abs(metrics[1]["recall"] - 1.0) < 1e-6

    assert metrics[2]["relevant_hits"] == 1
    assert abs(metrics[2]["precision"] - 0.5) < 1e-6
    assert abs(metrics[2]["recall"] - 1.0) < 1e-6


def test_k_larger_than_retrieved():
    qa = {"expected_chunk_ids": "x"}
    retrieved = [{"chunk_id": "x"}]
    ks = [5]
    metrics = RetrievalEvaluator.compute_metrics_for_example(qa, retrieved, ks)

    # only 1 doc retrieved, so precision uses k_used=1 denominator
    assert metrics[5]["relevant_hits"] == 1
    assert abs(metrics[5]["precision"] - 1.0) < 1e-6
    assert abs(metrics[5]["recall"] - 1.0) < 1e-6
