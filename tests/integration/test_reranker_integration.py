from pathlib import Path

from src.ingestion.chunker import MarkdownChunker
from src.ingestion.embedder import OllamaEmbedder
from src.ingestion.loaders.markdown_loader import MarkdownLoader
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.semantic_search import SemanticSearcher
from src.vectorstore.repository import MilvusRepository
from src.retrieval.reranker import MetadataReranker

def test_real_reranker_integration(tmp_path):
    document_path = Path(
        "knowledge-base/kubernetes/content/en/docs/"
        "concepts/overview/components.md"
    )

    repository = MilvusRepository(
        collection_name="kubernetes_reranker_test",
        dimension=768,
        db_path=str(tmp_path / "milvus.db"),
    )

    embedder = OllamaEmbedder(
        model="nomic-embed-text",
    )

    chunker = MarkdownChunker(
        max_chars=2000,
    )

    loader = MarkdownLoader()

    pipeline = IngestionPipeline(
        repository=repository,
        embedder=embedder,
        chunker=chunker,
        loader=loader,
    )

    ingestion_result = pipeline.ingest_file(
        document_path,
    )

    assert ingestion_result["chunks"] > 0
    assert ingestion_result["embeddings"] > 0

    semantic_searcher = SemanticSearcher(
        repository=repository,
        embedder=embedder,
    )

    query = "What are the components of the Kubernetes control plane?"

    semantic_results = semantic_searcher.search(
        query=query,
        top_k=5,
    )

    assert len(semantic_results) > 0

    print()
    print("=" * 80)
    print("SEMANTIC SEARCH RESULTS")
    print("=" * 80)

    for index, result in enumerate(semantic_results, start=1):
        print()
        print(f"RESULT {index}")
        print(f"Heading: {result['heading']}")
        print(f"Semantic score: {result['distance']:.4f}")
        print(f"Chunk index: {result['chunk_index']}")
        print("-" * 80)
        print(result["content"][:500])

    reranker = MetadataReranker()

    reranked_results = reranker.rerank(
        query=query,
        results=semantic_results,
        top_k=3,
    )

    assert len(reranked_results) == 3

    for result in reranked_results:
        assert "rerank_score" in result

    print()
    print("=" * 80)
    print("RERANKED RESULTS")
    print("=" * 80)

    for index, result in enumerate(reranked_results, start=1):
        print()
        print(f"RESULT {index}")
        print(f"Heading: {result['heading']}")
        print(f"Semantic score: {result['distance']:.4f}")
        print(f"Rerank score: {result['rerank_score']:.4f}")
        print(f"Chunk index: {result['chunk_index']}")
        print("-" * 80)
        print(result["content"][:500])