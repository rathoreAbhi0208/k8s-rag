from pathlib import Path

from src.ingestion.chunker import MarkdownChunker
from src.ingestion.embedder import OllamaEmbedder
from src.ingestion.loaders.markdown_loader import MarkdownLoader
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.hybrid_search import HybridSearcher
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


def test_real_hybrid_search(tmp_path):
    repository = MilvusRepository(
        collection_name="kubernetes_hybrid_search_test",
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

    semantic_searcher = SemanticSearcher(
        repository=repository,
        embedder=embedder,
    )

    hybrid_searcher = HybridSearcher(
        semantic_searcher=semantic_searcher,
        semantic_weight=0.7,
        keyword_weight=0.3,
    )

    query = "What are the components of the Kubernetes control plane?"

    results = hybrid_searcher.search(
        query=query,
        top_k=3,
        candidate_k=6,
    )

    print("\n" + "=" * 80)
    print("QUERY:")
    print(query)
    print("=" * 80)

    for i, result in enumerate(results, start=1):
        print(f"\nRESULT {i}")
        print(f"Heading: {result['heading']}")
        print(f"Semantic score: {result['semantic_score']:.4f}")
        print(f"Keyword score:  {result['keyword_score']:.4f}")
        print(f"Hybrid score:   {result['hybrid_score']:.4f}")
        print(f"Chunk index:    {result['chunk_index']}")
        print("-" * 80)
        print(result["content"])

    print("\n" + "=" * 80)

    assert len(results) > 0
    assert len(results) <= 3

    for result in results:
        assert "content" in result
        assert "heading" in result
        assert "semantic_score" in result
        assert "keyword_score" in result
        assert "hybrid_score" in result

    assert all(
        0.0 <= result["hybrid_score"] <= 1.0
        for result in results
    )