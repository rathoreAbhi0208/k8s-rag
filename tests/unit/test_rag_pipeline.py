from unittest.mock import Mock

import pytest

from src.rag.pipeline import RAGPipeline


def test_rag_pipeline_retrieves_and_generates():
    retriever = Mock()
    generator = Mock()

    retrieved_results = [
        {
            "id": 1,
            "heading": "Control Plane Components",
            "content": "The API server exposes the Kubernetes API.",
        }
    ]

    retriever.retrieve.return_value = retrieved_results

    generator.generate.return_value = (
        "The API server exposes the Kubernetes API."
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
    )

    result = pipeline.ask(
        "What does the API server do?"
    )

    assert result["query"] == "What does the API server do?"

    assert result["answer"] == (
        "The API server exposes the Kubernetes API."
    )

    assert result["sources"] == retrieved_results

    retriever.retrieve.assert_called_once_with(
        "What does the API server do?"
    )

    generator.generate.assert_called_once_with(
        query="What does the API server do?",
        results=retrieved_results,
    )


def test_empty_query_raises_error():
    retriever = Mock()
    generator = Mock()

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
    )

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        pipeline.ask("   ")

    retriever.retrieve.assert_not_called()
    generator.generate.assert_not_called()


def test_pipeline_handles_no_retrieved_results():
    retriever = Mock()
    generator = Mock()

    retriever.retrieve.return_value = []

    generator.generate.return_value = (
        "I don't have enough information in the provided context."
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
    )

    result = pipeline.ask(
        "What is something unknown?"
    )

    assert result["sources"] == []

    assert result["answer"] == (
        "I don't have enough information in the provided context."
    )

    generator.generate.assert_called_once_with(
        query="What is something unknown?",
        results=[],
    )