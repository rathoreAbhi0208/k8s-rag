from typing import Any

from src.ingestion.hasher import generate_chunk_id
from src.vectorstore.milvus_client import LocalMilvusClient


class MilvusRepository:
    """Store and search document chunks in Milvus."""

    def __init__(
        self,
        collection_name: str,
        dimension: int,
        db_path: str = "data/milvus.db",
    ):
        self.collection_name = collection_name

        self.milvus = LocalMilvusClient(
            db_path=db_path,
        )

        self.milvus.create_kubernetes_collection(
            collection_name=collection_name,
            dimension=dimension,
        )

    @property
    def client(self):
        return self.milvus.client

    def insert_chunks(
        self,
        chunks: list[Any],
        embeddings: list[list[float]],
    ) -> list[int]:
        """Insert document chunks and their embeddings."""

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings"
            )

        if not chunks:
            return []

        rows = []

        for chunk, embedding in zip(
            chunks,
            embeddings,
        ):
            metadata = chunk.metadata

            source = metadata.get(
                "source",
                "",
            )

            chunk_index = metadata.get(
                "chunk_index",
                0,
            )

            chunk_id = generate_chunk_id(
                source=source,
                chunk_index=chunk_index,
            )

            rows.append(
                {
                    "id": chunk_id,
                    "content": chunk.content,
                    "source": source,
                    "filename": metadata.get(
                        "filename",
                        "",
                    ),
                    "heading": metadata.get(
                        "heading",
                        "",
                    ),
                    "chunk_index": chunk_index,
                    "embedding": embedding,
                }
            )

        result = self.client.insert(
            collection_name=self.collection_name,
            data=rows,
        )

        return result["ids"]

    def count(self) -> int:
        """Return the number of entities in the collection."""

        result = self.client.query(
            collection_name=self.collection_name,
            filter="",
            output_fields=["id"],
            limit=10000,
        )

        return len(result)