from unittest.mock import Mock

import pytest

from src.generation.generator import Generator


def test_generator_builds_prompt_and_generates_answer():
    ollama_client = Mock()
    prompt_builder = Mock()

    prompt_builder.build.return_value = (
        "Generated RAG prompt"
    )

    ollama_client.generate.return_value = (
        "Kubernetes has a control plane and worker nodes."
    )

    generator = Generator(
        ollama_client=ollama_client,
        prompt_builder=prompt_builder,
    )

    results = [
        {
            "heading": "Core Components",
            "content": "Kubernetes has a control plane.",
        }
    ]

    answer = generator.generate(
        query="What is Kubernetes?",
        results=results,
    )

    assert answer == (
        "Kubernetes has a control plane and worker nodes."
    )

    prompt_builder.build.assert_called_once_with(
        query="What is Kubernetes?",
        results=results,
    )

    ollama_client.generate.assert_called_once_with(
        "Generated RAG prompt"
    )


def test_empty_query_raises_error():
    ollama_client = Mock()
    prompt_builder = Mock()

    generator = Generator(
        ollama_client=ollama_client,
        prompt_builder=prompt_builder,
    )

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        generator.generate(
            query="   ",
            results=[],
        )

    prompt_builder.build.assert_not_called()
    ollama_client.generate.assert_not_called()


def test_generator_handles_empty_results():
    ollama_client = Mock()
    prompt_builder = Mock()

    prompt_builder.build.return_value = (
        "No context prompt"
    )

    ollama_client.generate.return_value = (
        "I don't have enough information in the provided context."
    )

    generator = Generator(
        ollama_client=ollama_client,
        prompt_builder=prompt_builder,
    )

    answer = generator.generate(
        query="Unknown question",
        results=[],
    )

    assert answer == (
        "I don't have enough information in the provided context."
    )

    prompt_builder.build.assert_called_once_with(
        query="Unknown question",
        results=[],
    )