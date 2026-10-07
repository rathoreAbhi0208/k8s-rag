from unittest.mock import Mock

import pytest

from src.retrieval.retriever import Retriever


def make_result(result_id, heading, score):
    return {
        "id": result_id,
        "heading": heading,
        "content": f"Content for {heading}",
        "source": "test.md",
        "filename": "test.md",
        "chunk_index": result_id,
        "distance": score,
        "rerank_score": score,
        "hierarchy_expanded": False,
    }


def test_retrieve_runs_semantic_search_then_reranking():
    semantic_searcher = Mock()
    reranker = Mock()

    semantic_results = [
        make_result(1, "Core Components", 0.8),
        make_result(2, "Control Plane Components", 0.7),
        make_result(3, "Node Components", 0.6),
    ]

    reranked_results = [
        make_result(2, "Control Plane Components", 0.9),
        make_result(1, "Core Components", 0.8),
    ]

    semantic_searcher.search.return_value = semantic_results
    reranker.rerank.return_value = reranked_results

    retriever = Retriever(
        semantic_searcher=semantic_searcher,
        reranker=reranker,
        candidate_k=10,
        top_k=2,
    )

    results = retriever.retrieve(
        "What are the control plane components?"
    )

    assert results == reranked_results

    semantic_searcher.search.assert_called_once_with(
        query="What are the control plane components?",
        top_k=10,
    )

    reranker.rerank.assert_called_once_with(
        query="What are the control plane components?",
        results=semantic_results,
        top_k=2,
    )


def test_empty_query_raises_error():
    semantic_searcher = Mock()
    reranker = Mock()

    retriever = Retriever(
        semantic_searcher=semantic_searcher,
        reranker=reranker,
    )

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        retriever.retrieve("   ")


def test_empty_semantic_results_returns_empty():
    semantic_searcher = Mock()
    reranker = Mock()

    semantic_searcher.search.return_value = []

    retriever = Retriever(
        semantic_searcher=semantic_searcher,
        reranker=reranker,
    )

    results = retriever.retrieve("control plane")

    assert results == []

    reranker.rerank.assert_not_called()


def test_invalid_candidate_k():
    semantic_searcher = Mock()
    reranker = Mock()

    with pytest.raises(
        ValueError,
        match="candidate_k must be greater than 0",
    ):
        Retriever(
            semantic_searcher=semantic_searcher,
            reranker=reranker,
            candidate_k=0,
        )


def test_invalid_top_k():
    semantic_searcher = Mock()
    reranker = Mock()

    with pytest.raises(
        ValueError,
        match="top_k must be greater than 0",
    ):
        Retriever(
            semantic_searcher=semantic_searcher,
            reranker=reranker,
            top_k=0,
        )


def test_top_k_cannot_exceed_candidate_k():
    semantic_searcher = Mock()
    reranker = Mock()

    with pytest.raises(
        ValueError,
        match="top_k cannot be greater than candidate_k",
    ):
        Retriever(
            semantic_searcher=semantic_searcher,
            reranker=reranker,
            candidate_k=3,
            top_k=5,
        )