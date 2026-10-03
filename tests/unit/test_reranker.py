from unittest.mock import Mock, patch

import pytest

from src.retrieval.reranker import CrossEncoderReranker


def make_result(result_id, heading, content):
    return {
        "id": result_id,
        "distance": 0.8,
        "content": content,
        "source": "test.md",
        "filename": "test.md",
        "heading": heading,
        "chunk_index": result_id,
    }


@patch("src.retrieval.reranker.CrossEncoder")
def test_reranker_returns_ranked_results(mock_cross_encoder):
    mock_model = Mock()

    mock_model.predict.return_value = [
        0.2,
        0.9,
        0.5,
    ]

    mock_cross_encoder.return_value = mock_model

    reranker = CrossEncoderReranker()

    results = [
        make_result(
            1,
            "Node Components",
            "Worker nodes run workloads.",
        ),
        make_result(
            2,
            "Control Plane Components",
            "The control plane manages the cluster.",
        ),
        make_result(
            3,
            "Addons",
            "Addons extend Kubernetes functionality.",
        ),
    ]

    ranked = reranker.rerank(
        query="What manages the Kubernetes cluster?",
        results=results,
        top_k=3,
    )

    assert len(ranked) == 3

    assert ranked[0]["id"] == 2
    assert ranked[1]["id"] == 3
    assert ranked[2]["id"] == 1

    assert ranked[0]["rerank_score"] == 0.9


@patch("src.retrieval.reranker.CrossEncoder")
def test_reranker_limits_top_k(mock_cross_encoder):
    mock_model = Mock()

    mock_model.predict.return_value = [
        0.2,
        0.9,
        0.5,
    ]

    mock_cross_encoder.return_value = mock_model

    reranker = CrossEncoderReranker()

    results = [
        make_result(1, "A", "Document A"),
        make_result(2, "B", "Document B"),
        make_result(3, "C", "Document C"),
    ]

    ranked = reranker.rerank(
        query="test query",
        results=results,
        top_k=2,
    )

    assert len(ranked) == 2
    assert ranked[0]["id"] == 2
    assert ranked[1]["id"] == 3


@patch("src.retrieval.reranker.CrossEncoder")
def test_reranker_passes_query_document_pairs(mock_cross_encoder):
    mock_model = Mock()

    mock_model.predict.return_value = [
        0.8,
        0.4,
    ]

    mock_cross_encoder.return_value = mock_model

    reranker = CrossEncoderReranker()

    results = [
        make_result(1, "A", "Document A"),
        make_result(2, "B", "Document B"),
    ]

    reranker.rerank(
        query="my query",
        results=results,
        top_k=2,
    )

    mock_model.predict.assert_called_once_with(
        [
            ("my query", "Document A"),
            ("my query", "Document B"),
        ]
    )


@patch("src.retrieval.reranker.CrossEncoder")
def test_empty_results_returns_empty(mock_cross_encoder):
    reranker = CrossEncoderReranker()

    results = reranker.rerank(
        query="test query",
        results=[],
    )

    assert results == []


@patch("src.retrieval.reranker.CrossEncoder")
def test_empty_query_raises_error(mock_cross_encoder):
    reranker = CrossEncoderReranker()

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        reranker.rerank(
            query="   ",
            results=[],
        )


@patch("src.retrieval.reranker.CrossEncoder")
def test_invalid_top_k_raises_error(mock_cross_encoder):
    reranker = CrossEncoderReranker()

    with pytest.raises(
        ValueError,
        match="top_k must be greater than 0",
    ):
        reranker.rerank(
            query="test query",
            results=[],
            top_k=0,
        )