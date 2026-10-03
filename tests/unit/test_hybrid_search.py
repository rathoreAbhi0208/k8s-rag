from unittest.mock import Mock

import pytest

from src.retrieval.hybrid_search import HybridSearcher


def make_result(
    result_id,
    heading,
    content,
    distance,
):
    return {
        "id": result_id,
        "distance": distance,
        "content": content,
        "source": "test.md",
        "filename": "test.md",
        "heading": heading,
        "chunk_index": result_id,
    }


def test_hybrid_search_returns_results():
    semantic_searcher = Mock()

    semantic_searcher.search.return_value = [
        make_result(
            1,
            "Control Plane Components",
            "The control plane manages the cluster.",
            0.9,
        ),
        make_result(
            2,
            "Node Components",
            "Worker nodes run workloads.",
            0.7,
        ),
    ]

    semantic_searcher.repository.get_all_chunks.return_value = [
        make_result(
            1,
            "Control Plane Components",
            "The control plane manages the cluster.",
            0.9,
        ),
        make_result(
            2,
            "Node Components",
            "Worker nodes run workloads.",
            0.7,
        ),
    ]

    searcher = HybridSearcher(
        semantic_searcher=semantic_searcher,
    )

    results = searcher.search(
        query="control plane components",
        top_k=2,
    )

    assert len(results) == 2

    assert "hybrid_score" in results[0]
    assert "semantic_score" in results[0]
    assert "keyword_score" in results[0]

    assert results[0]["heading"] == "Control Plane Components"


def test_empty_query_raises_error():
    semantic_searcher = Mock()

    searcher = HybridSearcher(
        semantic_searcher=semantic_searcher,
    )

    with pytest.raises(ValueError, match="Query cannot be empty"):
        searcher.search("   ")


def test_invalid_top_k_raises_error():
    semantic_searcher = Mock()

    searcher = HybridSearcher(
        semantic_searcher=semantic_searcher,
    )

    with pytest.raises(
        ValueError,
        match="top_k must be greater than 0",
    ):
        searcher.search(
            "control plane",
            top_k=0,
        )


def test_invalid_candidate_k_raises_error():
    semantic_searcher = Mock()

    searcher = HybridSearcher(
        semantic_searcher=semantic_searcher,
    )

    with pytest.raises(
        ValueError,
        match="candidate_k must be greater than 0",
    ):
        searcher.search(
            "control plane",
            candidate_k=0,
        )


def test_weights_cannot_both_be_zero():
    semantic_searcher = Mock()

    with pytest.raises(
        ValueError,
        match="At least one search weight",
    ):
        HybridSearcher(
            semantic_searcher=semantic_searcher,
            semantic_weight=0,
            keyword_weight=0,
        )


def test_scores_are_normalized():
    scores = [0.2, 0.5, 0.8]

    normalized = HybridSearcher._normalize_scores(scores)

    assert normalized == pytest.approx([0.0, 0.5, 1.0])


def test_equal_scores_are_normalized_to_one():
    scores = [0.5, 0.5, 0.5]

    normalized = HybridSearcher._normalize_scores(scores)

    assert normalized == [1.0, 1.0, 1.0]


def test_tokenization():
    tokens = HybridSearcher._tokenize(
        "Kubernetes Control-Plane Components!"
    )

    assert tokens == [
        "kubernetes",
        "control",
        "plane",
        "components",
    ]

def test_bm25_can_add_documents_outside_semantic_results():
    semantic_results = [
        {
            "id": 1,
            "distance": 0.90,
            "content": "Kubernetes components",
            "source": "components.md",
            "filename": "components.md",
            "heading": "Core Components",
            "chunk_index": 0,
        }
    ]

    corpus = [
        {
            "id": 1,
            "content": "Kubernetes components",
            "source": "components.md",
            "filename": "components.md",
            "heading": "Core Components",
            "chunk_index": 0,
        },
        {
            "id": 2,
            "content": "kube-apiserver handles Kubernetes API requests",
            "source": "components.md",
            "filename": "components.md",
            "heading": "Control Plane Components",
            "chunk_index": 1,
        },
    ]

    class FakeSemanticSearcher:
        repository = type(
            "Repository",
            (),
            {
                "get_all_chunks": lambda self: corpus,
            },
        )()

        def search(self, query, top_k):
            return semantic_results

    searcher = HybridSearcher(
        semantic_searcher=FakeSemanticSearcher(),
        semantic_weight=0.7,
        keyword_weight=0.3,
    )

    results = searcher.search(
        query="kube-apiserver",
        top_k=2,
        candidate_k=1,
    )

    result_ids = [result["id"] for result in results]

    assert 2 in result_ids
    assert len(results) == 2