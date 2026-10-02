from unittest.mock import Mock

import pytest

from src.retrieval.semantic_search import SemanticSearcher


def test_search_returns_results():
    repository = Mock()
    embedder = Mock()

    embedder.embed.return_value = [0.1, 0.2, 0.3]

    repository.collection_name = "kubernetes_docs"

    repository.client.search.return_value = [
        [
            {
                "id": 123,
                "distance": 0.95,
                "entity": {
                    "content": "The Kubernetes control plane manages the cluster.",
                    "source": "components.md",
                    "filename": "components.md",
                    "heading": "Control Plane Components",
                    "chunk_index": 2,
                },
            }
        ]
    ]

    searcher = SemanticSearcher(
        repository=repository,
        embedder=embedder,
    )

    results = searcher.search(
        query="What manages the Kubernetes cluster?",
        top_k=3,
    )

    assert len(results) == 1
    assert results[0]["id"] == 123
    assert results[0]["distance"] == 0.95
    assert results[0]["heading"] == "Control Plane Components"
    assert results[0]["chunk_index"] == 2

    embedder.embed.assert_called_once_with(
        "What manages the Kubernetes cluster?"
    )

    repository.client.search.assert_called_once_with(
        collection_name="kubernetes_docs",
        data=[[0.1, 0.2, 0.3]],
        anns_field="embedding",
        limit=3,
        output_fields=[
            "content",
            "source",
            "filename",
            "heading",
            "chunk_index",
        ],
    )


def test_empty_query_raises_error():
    repository = Mock()
    embedder = Mock()

    searcher = SemanticSearcher(
        repository=repository,
        embedder=embedder,
    )

    with pytest.raises(ValueError, match="Query cannot be empty"):
        searcher.search("   ")


def test_invalid_top_k_raises_error():
    repository = Mock()
    embedder = Mock()

    searcher = SemanticSearcher(
        repository=repository,
        embedder=embedder,
    )

    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        searcher.search("What is Kubernetes?", top_k=0)


def test_multiple_results_are_returned():
    repository = Mock()
    embedder = Mock()

    embedder.embed.return_value = [0.1, 0.2]

    repository.collection_name = "kubernetes_docs"

    repository.client.search.return_value = [
        [
            {
                "id": 1,
                "distance": 0.98,
                "entity": {
                    "content": "Kubernetes is a container orchestration platform.",
                    "source": "intro.md",
                    "filename": "intro.md",
                    "heading": "Introduction",
                    "chunk_index": 0,
                },
            },
            {
                "id": 2,
                "distance": 0.91,
                "entity": {
                    "content": "The control plane manages the cluster.",
                    "source": "components.md",
                    "filename": "components.md",
                    "heading": "Control Plane",
                    "chunk_index": 1,
                },
            },
        ]
    ]

    searcher = SemanticSearcher(
        repository=repository,
        embedder=embedder,
    )

    results = searcher.search(
        query="How does Kubernetes work?",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["id"] == 1
    assert results[1]["id"] == 2