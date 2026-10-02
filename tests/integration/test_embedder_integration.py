from src.ingestion.embedder import OllamaEmbedder


MODEL = "nomic-embed-text"
EXPECTED_DIMENSION = 768


def test_real_ollama_embedding():
    embedder = OllamaEmbedder(
        model=MODEL
    )

    text = (
        "Kubernetes control plane manages "
        "the overall state of a cluster."
    )

    embedding = embedder.embed(text)

    assert isinstance(
        embedding,
        list,
    )

    assert len(embedding) == EXPECTED_DIMENSION

    assert all(
        isinstance(value, float)
        for value in embedding
    )


def test_real_ollama_batch_embedding():
    embedder = OllamaEmbedder(
        model=MODEL
    )

    texts = [
        "Kubernetes control plane",
        "Kubernetes worker node",
    ]

    embeddings = embedder.embed_batch(
        texts
    )

    assert len(embeddings) == len(texts)

    for embedding in embeddings:
        assert len(embedding) == EXPECTED_DIMENSION
        assert all(
            isinstance(value, float)
            for value in embedding
        )