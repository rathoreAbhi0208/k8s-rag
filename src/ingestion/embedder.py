import ollama


class OllamaEmbedder:
    """Generate embeddings using a local Ollama embedding model."""

    def __init__(
        self,
        model: str = "nomic-embed-text:latest",
    ):
        self.model = model

    def embed(self, text: str) -> list[float]:
        """Generate an embedding for a single piece of text."""

        if not text.strip():
            raise ValueError("Cannot embed empty text")

        response = ollama.embed(
            model=self.model,
            input=text,
        )

        return response["embeddings"][0]

    def embed_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts."""

        if not texts:
            return []

        response = ollama.embed(
            model=self.model,
            input=texts,
        )

        return response["embeddings"]

