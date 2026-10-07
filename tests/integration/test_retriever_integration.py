from pathlib import Path

from src.ingestion.chunker import MarkdownChunker
from src.ingestion.embedder import OllamaEmbedder
from src.ingestion.loaders.markdown_loader import MarkdownLoader
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.retriever import Retriever
from src.retrieval.semantic_search import SemanticSearcher
from src.vectorstore.repository import MilvusRepository
from src.retrieval.reranker import MetadataReranker


def test_real_retriever_integration(tmp_path):
    document_path = Path(
        "knowledge-base/kubernetes/content/en/docs/"
        "concepts/overview/components.md"
    )

    repository = MilvusRepository(
        collection_name="kubernetes_retriever_test",
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

    ingestion_result = pipeline.ingest_file(
        document_path,
    )

    assert ingestion_result["chunks"] > 0

    semantic_searcher = SemanticSearcher(
        repository=repository,
        embedder=embedder,
    )

    reranker = MetadataReranker()

    retriever = Retriever(
        semantic_searcher=semantic_searcher,
        reranker=reranker,
        candidate_k=20,
        top_k=3,
        use_reranker=True,
        use_hierarchy=True,
    )

    query = "What are the components of the Kubernetes control plane?"

    results = retriever.retrieve(query)

    assert len(results) == 3

    for result in results:
        assert "content" in result
        assert "heading" in result
        assert "distance" in result
        assert "rerank_score" in result

    print()
    print("=" * 80)
    print("FINAL RETRIEVER RESULTS")
    print("=" * 80)

    for index, result in enumerate(results, start=1):
        print()
        print(f"RESULT {index}")
        print(f"Heading: {result['heading']}")
        print(f"Semantic score: {result['distance']:.4f}")
        print(f"Rerank score: {result['rerank_score']:.4f}")
        print(f"Chunk index: {result['chunk_index']}")
        print("-" * 80)
        print(result["content"][:500])