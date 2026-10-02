from unittest.mock import patch

import pytest

from src.ingestion.embedder import OllamaEmbedder


def test_embed_returns_embedding():
    fake_embedding = [0.1, 0.2, 0.3]

    fake_response = {
        "embeddings": [
            fake_embedding
        ]
    }

    with patch(
        "src.ingestion.embedder.ollama.embed",
        return_value=fake_response,
    ) as mock_embed:

        embedder = OllamaEmbedder(
            model="nomic-embed-text"
        )

        result = embedder.embed(
            "Kubernetes control plane"
        )

    assert result == fake_embedding

    mock_embed.assert_called_once_with(
        model="nomic-embed-text",
        input="Kubernetes control plane",
    )


def test_embed_rejects_empty_text():
    embedder = OllamaEmbedder(
        model="nomic-embed-text"
    )

    with pytest.raises(ValueError):
        embedder.embed("")


def test_embed_rejects_whitespace():
    embedder = OllamaEmbedder(
        model="nomic-embed-text"
    )

    with pytest.raises(ValueError):
        embedder.embed("   ")


def test_embed_batch_returns_embeddings():
    fake_embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    fake_response = {
        "embeddings": fake_embeddings
    }

    with patch(
        "src.ingestion.embedder.ollama.embed",
        return_value=fake_response,
    ) as mock_embed:

        embedder = OllamaEmbedder(
            model="nomic-embed-text"
        )

        texts = [
            "Kubernetes control plane",
            "Kubernetes worker node",
        ]

        result = embedder.embed_batch(texts)

    assert result == fake_embeddings

    mock_embed.assert_called_once_with(
        model="nomic-embed-text",
        input=texts,
    )


def test_embed_batch_empty_list():
    embedder = OllamaEmbedder(
        model="nomic-embed-text"
    )

    result = embedder.embed_batch([])

    assert result == []