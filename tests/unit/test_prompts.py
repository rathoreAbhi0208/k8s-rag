import pytest

from src.generation.prompts import PromptBuilder


def test_build_prompt_contains_query():
    builder = PromptBuilder()

    prompt = builder.build(
        query="What are Kubernetes control plane components?",
        results=[],
    )

    assert "What are Kubernetes control plane components?" in prompt


def test_build_prompt_contains_retrieved_context():
    builder = PromptBuilder()

    results = [
        {
            "heading": "Control Plane Components",
            "content": "The API server exposes the Kubernetes API.",
        },
        {
            "heading": "Node Components",
            "content": "The kubelet runs on every node.",
        },
    ]

    prompt = builder.build(
        query="What does the API server do?",
        results=results,
    )

    assert "Control Plane Components" in prompt
    assert "The API server exposes the Kubernetes API." in prompt
    assert "Node Components" in prompt
    assert "The kubelet runs on every node." in prompt


def test_build_prompt_contains_grounding_instruction():
    builder = PromptBuilder()

    prompt = builder.build(
        query="What is Kubernetes?",
        results=[],
    )

    assert "using only the provided context" in prompt
    assert "Do not invent or assume information" in prompt


def test_empty_results_are_handled():
    builder = PromptBuilder()

    prompt = builder.build(
        query="What is Kubernetes?",
        results=[],
    )

    assert "No context was retrieved." in prompt


def test_empty_query_raises_error():
    builder = PromptBuilder()

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        builder.build(
            query="   ",
            results=[],
        )


def test_context_preserves_document_order():
    builder = PromptBuilder()

    results = [
        {
            "heading": "First",
            "content": "First document",
        },
        {
            "heading": "Second",
            "content": "Second document",
        },
    ]

    prompt = builder.build(
        query="test",
        results=results,
    )

    first_position = prompt.index("First document")
    second_position = prompt.index("Second document")

    assert first_position < second_position