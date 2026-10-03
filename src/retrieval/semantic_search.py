from typing import Any

from src.ingestion.embedder import OllamaEmbedder
from src.vectorstore.repository import MilvusRepository


class SemanticSearcher:
    """Perform semantic vector search against the Milvus collection."""

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
        """Search the vector database using semantic similarity."""

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        # ---------------------------------------------------------
        # Generate query embedding
        # ---------------------------------------------------------

        query_embedding = self.embedder.embed(query)

        # ---------------------------------------------------------
        # Vector search
        # ---------------------------------------------------------

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
                "heading_level",
                "heading_path",
            ],
        )

        # ---------------------------------------------------------
        # Convert Milvus results
        # ---------------------------------------------------------

        matches: list[dict[str, Any]] = []

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
                    "heading_level": entity["heading_level"],
                    "heading_path": entity["heading_path"],
                }
            )

        # # ---------------------------------------------------------
        # # Debug output
        # # ---------------------------------------------------------

        # print("\n" + "=" * 80)
        # print("SEMANTIC SEARCH RESULTS")
        # print("=" * 80)

        # print(f"Query: {query}")
        # print(f"Top K: {top_k}")

        # for index, match in enumerate(matches, start=1):
        #     print(
        #         f"\n{index}. "
        #         f"distance={match['distance']:.4f}"
        #     )
        #     print(
        #         f"   Heading:      {match['heading']}"
        #     )
        #     print(
        #         f"   Heading Path: {match['heading_path']}"
        #     )
        #     print(
        #         f"   Source:       {match['source']}"
        #     )
        #     print(
        #         f"   Chunk Index:  {match['chunk_index']}"
        #     )

        # print("=" * 80)

        return matches