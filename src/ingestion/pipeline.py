from pathlib import Path

from src.ingestion.chunker import MarkdownChunker
from src.ingestion.embedder import OllamaEmbedder
from src.ingestion.loaders.markdown_loader import MarkdownLoader
from src.vectorstore.repository import MilvusRepository


class IngestionPipeline:
    """Load, chunk, embed, and store documents."""

    def __init__(
        self,
        repository: MilvusRepository,
        embedder: OllamaEmbedder,
        chunker: MarkdownChunker | None = None,
        loader: MarkdownLoader | None = None,
    ):
        self.repository = repository
        self.embedder = embedder
        self.chunker = chunker or MarkdownChunker()
        self.loader = loader or MarkdownLoader()

    def ingest_file(
        self,
        file_path: str | Path,
    ) -> dict:
        """Ingest a single Markdown document."""

        path = Path(file_path)

        # Load document
        document = self.loader.load(path)

        # Create chunks
        chunks = self.chunker.chunk(
            content=document["content"],
            metadata=document["metadata"],
        )

        if not chunks:
            return {
                "source": str(path),
                "chunks": 0,
                "embeddings": 0,
                "inserted_ids": [],
            }

        # Generate embeddings
        texts = [
            (
                f"Source: {chunk.metadata.get('source', '')}\n"
                f"Document: {chunk.metadata.get('filename', '')}\n"
                f"Section: {chunk.metadata.get('heading_path', '')}\n\n"
                f"{chunk.content}"
            )
            for chunk in chunks
        ]

        embeddings = self.embedder.embed_batch(
            texts
        )

        # Store in Milvus
        inserted_ids = self.repository.insert_chunks(
            chunks=chunks,
            embeddings=embeddings,
        )

        return {
            "source": str(path),
            "chunks": len(chunks),
            "embeddings": len(embeddings),
            "inserted_ids": inserted_ids,
        }