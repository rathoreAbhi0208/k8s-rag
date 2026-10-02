from pathlib import Path

from src.ingestion.chunker import MarkdownChunker
from src.ingestion.embedder import OllamaEmbedder
from src.ingestion.loaders.markdown_loader import MarkdownLoader
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.semantic_search import SemanticSearcher
from src.vectorstore.repository import MilvusRepository


PROJECT_ROOT = Path(__file__).resolve().parents[2]

COMPONENTS_FILE = (
    PROJECT_ROOT
    / "knowledge-base"
    / "kubernetes"
    / "content"
    / "en"
    / "docs"
    / "concepts"
    / "overview"
    / "components.md"
)


def test_real_semantic_search(tmp_path):
    repository = MilvusRepository(
        collection_name="kubernetes_search_test",
        dimension=768,
        db_path=str(tmp_path / "milvus.db"),
    )

    embedder = OllamaEmbedder(
        model="nomic-embed-text",
    )

    pipeline = IngestionPipeline(
        repository=repository,
        embedder=embedder,
        chunker=MarkdownChunker(max_chars=2000),
        loader=MarkdownLoader(),
    )

    ingestion_result = pipeline.ingest_file(COMPONENTS_FILE)

    assert ingestion_result["chunks"] > 0
    assert len(ingestion_result["inserted_ids"]) == ingestion_result["chunks"]

    searcher = SemanticSearcher(
        repository=repository,
        embedder=embedder,
    )

    results = searcher.search(
        query="What are the components of the Kubernetes control plane?",
        top_k=3,
    )

    print("\n" + "=" * 80)
    print("QUERY:")
    print("What are the components of the Kubernetes control plane?")
    print("=" * 80)

    for i, result in enumerate(results, start=1):
        print(f"\nRESULT {i}")
        print(f"Score: {result['distance']}")
        print(f"Heading: {result['heading']}")
        print(f"Source: {result['source']}")
        print(f"Chunk index: {result['chunk_index']}")
        print("-" * 80)
        print(result["content"])

    print("\n" + "=" * 80)

    assert len(results) > 0
    assert len(results) <= 3

    for result in results:
        assert "content" in result
        assert "source" in result
        assert "heading" in result
        assert "distance" in result

    headings = [
    result["heading"].lower()
    for result in results
    ]

    assert len(results) == 3
    assert all(result["content"].strip() for result in results)
    assert all(result["source"].endswith(".md") for result in results)