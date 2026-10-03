from pathlib import Path

from src.ingestion.embedder import OllamaEmbedder
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.chunker import MarkdownChunker
from src.ingestion.loaders.markdown_loader import MarkdownLoader
from src.vectorstore.repository import MilvusRepository


KNOWLEDGE_BASE = Path("knowledge-base/kubernetes/content/en/docs")
MILVUS_DB = "data/milvus.db"
COLLECTION_NAME = "kubernetes_docs"
EMBEDDING_DIMENSION = 768


def main():
    if not KNOWLEDGE_BASE.exists():
        raise FileNotFoundError(
            f"Kubernetes documentation directory not found: {KNOWLEDGE_BASE}"
        )

    embedder = OllamaEmbedder(
        model="nomic-embed-text",
    )

    repository = MilvusRepository(
        collection_name=COLLECTION_NAME,
        dimension=EMBEDDING_DIMENSION,
        db_path=MILVUS_DB,
    )

    pipeline = IngestionPipeline(
        repository=repository,
        embedder=embedder,
        chunker=MarkdownChunker(max_chars=2000),
        loader=MarkdownLoader(),
    )

    files = sorted(KNOWLEDGE_BASE.rglob("*.md"))

    print(f"Found {len(files)} Markdown files")

    total_chunks = 0
    total_embeddings = 0

    for index, file_path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {file_path}")

        result = pipeline.ingest_file(file_path)

        total_chunks += result["chunks"]
        total_embeddings += result["embeddings"]

    print()
    print("Ingestion completed")
    print(f"Files processed: {len(files)}")
    print(f"Chunks created: {total_chunks}")
    print(f"Embeddings created: {total_embeddings}")
    print(f"Milvus database: {MILVUS_DB}")
    print(f"Collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    main()