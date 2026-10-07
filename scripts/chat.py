from pathlib import Path
from src.retrieval.reranker import MetadataReranker
from src.generation.generator import Generator
from src.generation.ollama_client import OllamaClient
from src.generation.prompts import PromptBuilder
from src.ingestion.embedder import OllamaEmbedder
from src.rag.pipeline import RAGPipeline
from src.retrieval.retriever import Retriever
from src.retrieval.semantic_search import SemanticSearcher
from src.vectorstore.collections import EMBEDDING_DIMENSION, KUBERNETES_COLLECTION
from src.vectorstore.repository import MilvusRepository


MILVUS_DB = Path("data/milvus.db")
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"


def build_rag_pipeline() -> RAGPipeline:
    if not MILVUS_DB.exists():
        raise FileNotFoundError(
            f"Milvus database not found: {MILVUS_DB}\n"
            "Run the ingestion script first."
        )

    repository = MilvusRepository(
        collection_name=KUBERNETES_COLLECTION,
        dimension=EMBEDDING_DIMENSION,
        db_path=str(MILVUS_DB),
    )

    embedder = OllamaEmbedder(model=EMBEDDING_MODEL)

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

    ollama_client = OllamaClient(model=LLM_MODEL)

    generator = Generator(
        ollama_client=ollama_client,
        prompt_builder=PromptBuilder(),
    )

    return RAGPipeline(
        retriever=retriever,
        generator=generator,
    )


def print_sources(sources: list[dict]) -> None:
    if not sources:
        print("\nSources: None")
        return

    print("\nSources:")

    for index, source in enumerate(sources, start=1):
        print(f"\n[{index}]")
        print(f"  File:    {source.get('filename', '')}")
        print(f"  Heading: {source.get('heading_path', '')}")
        print(f"  Score:   {source.get('distance', 0.0):.4f}")
        print(f"  Content:\n{source.get('content', '')[:1000]}")


def main() -> None:
    print("=" * 80)
    print("KUBERNETES RAG ASSISTANT")
    print("=" * 80)
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"LLM model:       {LLM_MODEL}")
    print(f"Vector DB:       {MILVUS_DB}")
    print("=" * 80)

    try:
        rag = build_rag_pipeline()
    except Exception as exc:
        print(f"\nFailed to initialize RAG pipeline: {exc}")
        return

    print("\nReady.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            query = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting.")
            break

        if not query:
            continue

        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            result = rag.ask(query)

            print("\n" + "-" * 80)
            print("ANSWER")
            print("-" * 80)
            print(result["answer"])

            print_sources(result["sources"])

            print("\n" + "=" * 80 + "\n")

        except Exception as exc:
            print(f"\nError while processing question: {exc}\n")


if __name__ == "__main__":
    main()