from typing import Any

from src.ingestion.hasher import generate_chunk_id
from src.vectorstore.milvus_client import LocalMilvusClient


class MilvusRepository:
    def __init__(
        self,
        collection_name: str,
        dimension: int,
        db_path: str = "data/milvus.db",
    ):
        self.collection_name = collection_name

        self.milvus = LocalMilvusClient(db_path=db_path)

        self.milvus.create_kubernetes_collection(
            collection_name=collection_name,
            dimension=dimension,
        )

        self._ensure_loaded()

    @property
    def client(self):
        return self.milvus.client

    def _ensure_loaded(self) -> None:
        """Ensure the collection is loaded before read/search operations."""
        if not self.client.has_collection(self.collection_name):
            raise ValueError(
                f"Collection does not exist: {self.collection_name}"
            )

        self.client.load_collection(
            collection_name=self.collection_name
        )

    def insert_chunks(self, chunks, embeddings):
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings"
            )

        if not chunks:
            return []

        rows = []

        for chunk, embedding in zip(chunks, embeddings):
            metadata = chunk.metadata

            source = metadata.get("source", "")
            chunk_index = metadata.get("chunk_index", 0)

            chunk_id = generate_chunk_id(
                source=source,
                chunk_index=chunk_index,
            )

            rows.append(
                {
                    "id": chunk_id,
                    "content": chunk.content,
                    "source": source,
                    "filename": metadata.get("filename", ""),
                    "heading": metadata.get("heading", ""),
                    "chunk_index": chunk_index,
                    "embedding": embedding,
                    "heading_level": metadata.get("heading_level", 0),
                    "heading_path": metadata.get("heading_path", ""),
                }
            )

        result = self.client.insert(
            collection_name=self.collection_name,
            data=rows,
        )

        return result["ids"]

    def count(self):
        self._ensure_loaded()

        result = self.client.query(
            collection_name=self.collection_name,
            filter="",
            output_fields=["id"],
            limit=10000,
        )

        return len(result)

    def get_all_chunks(self) -> list[dict[str, Any]]:
        """Return all stored document chunks for lexical retrieval."""
        return self.client.query(
            collection_name=self.collection_name,
            filter="",
            output_fields=[
                "id",
                "content",
                "source",
                "filename",
                "heading",
                "chunk_index",
            ],
            limit=10000,
        )

    def get_chunks_by_heading_path(
        self,
        heading_path: str,
    ) -> list[dict[str, Any]]:
        """Return chunks whose heading path starts with the given path."""

        self._ensure_loaded()

        if not heading_path.strip():
            raise ValueError("heading_path cannot be empty")

        escaped_path = heading_path.replace("\\", "\\\\").replace('"', '\\"')

        results = self.client.query(
            collection_name=self.collection_name,
            filter=f'heading_path like "{escaped_path}%"',
            output_fields=[
                "id",
                "content",
                "source",
                "filename",
                "heading",
                "chunk_index",
                "heading_level",
                "heading_path",
            ],
            limit=100,
        )

        return results