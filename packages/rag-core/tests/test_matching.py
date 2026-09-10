from rag_core.evaluation.matching import expected_targets


def test_expected_chunk_ids_accept_semicolon_delimiters():
    targets = expected_targets({"expected_chunk_ids": "chunk-a;chunk-b"})

    assert targets == [
        {"chunk_id": "chunk-a"},
        {"chunk_id": "chunk-b"},
    ]


def test_expected_chunk_ids_accept_comma_delimiters():
    targets = expected_targets({"expected_chunk_ids": "chunk-a,chunk-b"})

    assert targets == [
        {"chunk_id": "chunk-a"},
        {"chunk_id": "chunk-b"},
    ]
