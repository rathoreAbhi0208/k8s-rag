from pathlib import Path

from src.generation.generator import Generator
from src.generation.ollama_client import OllamaClient
from src.generation.prompts import PromptBuilder
from src.ingestion.chunker import MarkdownChunker
from src.ingestion.embedder import OllamaEmbedder
from src.ingestion.loaders.markdown_loader import MarkdownLoader
from src.ingestion.pipeline import IngestionPipeline
from src.rag.pipeline import RAGPipeline
from src.retrieval.retriever import Retriever
from src.retrieval.semantic_search import SemanticSearcher
from src.vectorstore.repository import MilvusRepository
from src.retrieval.reranker import MetadataReranker


def test_real_rag_pipeline(tmp_path):
    document_path = Path(
        "knowledge-base/kubernetes/content/en/docs/"
        "concepts/overview/components.md"
    )

    repository = MilvusRepository(
        collection_name="kubernetes_rag_test",
        dimension=768,
        db_path=str(tmp_path / "milvus.db"),
    )

    embedder = OllamaEmbedder(
        model="nomic-embed-text",
    )

    ingestion_pipeline = IngestionPipeline(
        repository=repository,
        embedder=embedder,
        chunker=MarkdownChunker(max_chars=2000),
        loader=MarkdownLoader(),
    )

    ingestion_result = ingestion_pipeline.ingest_file(
        document_path,
    )

    assert ingestion_result["chunks"] > 0
    assert ingestion_result["embeddings"] > 0

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

    ollama_client = OllamaClient(
        model="llama3.2:3b",
    )

    prompt_builder = PromptBuilder()

    generator = Generator(
        ollama_client=ollama_client,
        prompt_builder=prompt_builder,
    )

    rag_pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
    )

    query = "What are the components of the Kubernetes control plane?"

    result = rag_pipeline.ask(query)

    assert result["query"] == query
    assert result["answer"]
    assert isinstance(result["answer"], str)

    assert len(result["sources"]) == 3

    for source in result["sources"]:
        assert source["id"]
        assert source["content"]
        assert source["source"]
        assert source["filename"]
        assert source["heading"]
        assert isinstance(source["chunk_index"], int)
        assert isinstance(source["heading_level"], int)
        assert source["heading_path"]

    print()
    print("=" * 80)
    print("RAG PIPELINE RESULT")
    print("=" * 80)

    print()
    print("QUESTION:")
    print(query)

    print()
    print("ANSWER:")
    print(result["answer"])

    print()
    print("SOURCES:")
    print("-" * 80)

    for index, source in enumerate(
        result["sources"],
        start=1,
    ):
        print(
            f"{index}. "
            f"{source['heading']} "
        )

    print("=" * 80)