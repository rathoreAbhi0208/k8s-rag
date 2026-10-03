from unittest.mock import patch

import pytest

from src.generation.ollama_client import OllamaClient


@patch("src.generation.ollama_client.ollama.generate")
def test_generate_returns_response(mock_generate):
    mock_generate.return_value = {
        "response": "Kubernetes has a control plane and worker nodes."
    }

    client = OllamaClient(
        model="llama3.2:3b",
    )

    result = client.generate(
        "What is Kubernetes?"
    )

    assert result == (
        "Kubernetes has a control plane and worker nodes."
    )

    mock_generate.assert_called_once_with(
        model="llama3.2:3b",
        prompt="What is Kubernetes?",
    )


@patch("src.generation.ollama_client.ollama.generate")
def test_generate_uses_configured_model(mock_generate):
    mock_generate.return_value = {
        "response": "Answer"
    }

    client = OllamaClient(
        model="qwen3:8b",
    )

    client.generate(
        "What is a pod?"
    )

    mock_generate.assert_called_once_with(
        model="qwen3:8b",
        prompt="What is a pod?",
    )


@patch("src.generation.ollama_client.ollama.generate")
def test_empty_prompt_raises_error(mock_generate):
    client = OllamaClient()

    with pytest.raises(
        ValueError,
        match="Prompt cannot be empty",
    ):
        client.generate("   ")

    mock_generate.assert_not_called()