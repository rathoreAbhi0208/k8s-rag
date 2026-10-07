import pytest

from src.retrieval.reranker import MetadataReranker


def test_reranker_rejects_empty_query():
    reranker = MetadataReranker()

    with pytest.raises(ValueError, match="Query cannot be empty"):
        reranker.rerank(query="", results=[])


def test_reranker_rejects_invalid_top_k():
    reranker = MetadataReranker()

    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        reranker.rerank(
            query="control plane",
            results=[],
            top_k=0,
        )


def test_reranker_returns_empty_for_empty_results():
    reranker = MetadataReranker()

    result = reranker.rerank(
        query="control plane components",
        results=[],
    )

    assert result == []


def test_reranker_adds_scores():
    reranker = MetadataReranker()

    results = [
        {
            "distance": 0.8,
            "heading": "Control Plane Components",
            "heading_path": "Core Components > Control Plane Components",
            "content": "The Kubernetes control plane manages the cluster.",
        }
    ]

    reranked = reranker.rerank(
        query="What are the control plane components?",
        results=results,
        top_k=1,
    )

    assert len(reranked) == 1

    result = reranked[0]

    assert "semantic_score" in result
    assert "heading_score" in result
    assert "path_score" in result
    assert "content_score" in result
    assert "rerank_score" in result


def test_reranker_prioritizes_relevant_heading():
    reranker = MetadataReranker()

    results = [
        {
            "distance": 0.80,
            "heading": "Synopsis",
            "heading_path": "Synopsis",
            "content": "Generate the manifests for the new control plane components.",
        },
        {
            "distance": 0.74,
            "heading": "Control Plane Components",
            "heading_path": "Core Components > Control Plane Components",
            "content": "The Kubernetes control plane consists of several components.",
        },
    ]

    reranked = reranker.rerank(
        query="What are the components of the Kubernetes control plane?",
        results=results,
        top_k=2,
    )

    assert reranked[0]["heading"] == "Control Plane Components"


def test_reranker_respects_top_k():
    reranker = MetadataReranker()

    results = [
        {
            "distance": 0.7,
            "heading": f"Section {i}",
            "heading_path": f"Section {i}",
            "content": "Kubernetes documentation.",
        }
        for i in range(5)
    ]

    reranked = reranker.rerank(
        query="Kubernetes",
        results=results,
        top_k=2,
    )

    assert len(reranked) == 2