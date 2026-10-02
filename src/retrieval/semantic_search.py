from typing import Any

from src.ingestion.embedder import OllamaEmbedder
from src.vectorstore.repository import MilvusRepository


class SemanticSearcher:
    """Perform semantic similarity search over stored document chunks."""

    def __init__(
        self,
        repository: MilvusRepository,
        embedder: OllamaEmbedder,
    ):
        self.repository = repository
        self.embedder = embedder

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search for the most semantically similar document chunks."""

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_embedding = self.embedder.embed(query)

        results = self.repository.client.search(
            collection_name=self.repository.collection_name,
            data=[query_embedding],
            anns_field="embedding",
            limit=top_k,
            output_fields=[
                "content",
                "source",
                "filename",
                "heading",
                "chunk_index",
            ],
        )

        matches = []

        for result in results[0]:
            entity = result["entity"]

            matches.append(
                {
                    "id": result["id"],
                    "distance": result["distance"],
                    "content": entity["content"],
                    "source": entity["source"],
                    "filename": entity["filename"],
                    "heading": entity["heading"],
                    "chunk_index": entity["chunk_index"],
                }
            )

        return matches