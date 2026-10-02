from pathlib import Path

from src.ingestion.chunker import MarkdownChunker
from src.ingestion.embedder import OllamaEmbedder
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.loaders.markdown_loader import MarkdownLoader
from src.vectorstore.collections import (
    EMBEDDING_DIMENSION,
    KUBERNETES_COLLECTION,
)
from src.vectorstore.repository import MilvusRepository


DOCS_PATH = Path(
    "knowledge-base/kubernetes/content/en/docs"
)

FILE_PATH = (
    DOCS_PATH
    / "concepts"
    / "overview"
    / "components.md"
)


def test_real_ingestion_pipeline(
    tmp_path: Path,
):
    """Test the complete ingestion pipeline."""

    db_path = tmp_path / "test_rag.db"

    repository = MilvusRepository(
        collection_name=KUBERNETES_COLLECTION,
        dimension=EMBEDDING_DIMENSION,
        db_path=str(db_path),
    )

    embedder = OllamaEmbedder(
        model="nomic-embed-text"
    )

    chunker = MarkdownChunker(
        max_chars=2000
    )

    loader = MarkdownLoader()

    pipeline = IngestionPipeline(
        repository=repository,
        embedder=embedder,
        chunker=chunker,
        loader=loader,
    )

    result = pipeline.ingest_file(
        FILE_PATH
    )

    # ---------------------------------------------------------
    # Verify ingestion result
    # ---------------------------------------------------------

    assert result["source"] == str(FILE_PATH)

    assert result["chunks"] > 0

    assert (
        result["embeddings"]
        == result["chunks"]
    )

    assert (
        len(result["inserted_ids"])
        == result["chunks"]
    )

    # ---------------------------------------------------------
    # Verify records exist in Milvus
    # ---------------------------------------------------------


    # ---------------------------------------------------------
    # Verify stored data
    # ---------------------------------------------------------

    records = repository.client.query(
        collection_name=KUBERNETES_COLLECTION,
        filter="",
        output_fields=[
            "id",
            "content",
            "source",
            "filename",
            "heading",
            "chunk_index",
        ],
    )

    assert len(records) == result["chunks"]

    for record in records:
        assert record["content"]
        assert record["source"]
        assert record["filename"]
        assert record["heading"]
        assert isinstance(
            record["chunk_index"],
            int,
        )

    # ---------------------------------------------------------
    # Verify embeddings through vector search
    # ---------------------------------------------------------

    query_text = (
        "What components are part of "
        "the Kubernetes control plane?"
    )

    query_embedding = embedder.embed(
        query_text
    )

    search_results = repository.client.search(
        collection_name=KUBERNETES_COLLECTION,
        data=[query_embedding],
        limit=3,
        output_fields=[
            "content",
            "heading",
            "source",
        ],
    )

    assert len(search_results) == 1

    assert len(search_results[0]) > 0

    top_result = search_results[0][0]

    assert top_result["entity"]["content"]
    assert top_result["entity"]["heading"]
    assert top_result["entity"]["source"]