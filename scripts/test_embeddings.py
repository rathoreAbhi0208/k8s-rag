from pathlib import Path

from src.ingestion.chunker import MarkdownChunker
from src.ingestion.embedder import OllamaEmbedder
from src.ingestion.loaders.markdown_loader import MarkdownLoader


DOCS_PATH = Path(
    "knowledge-base/kubernetes/content/en/docs"
)

FILE_PATH = (
    DOCS_PATH
    / "concepts"
    / "overview"
    / "components.md"
)


def main():
    print("\n" + "=" * 80)
    print("OLLAMA EMBEDDING TEST")
    print("=" * 80)

    # ---------------------------------------------------------
    # Load document
    # ---------------------------------------------------------

    loader = MarkdownLoader()

    document = loader.load(FILE_PATH)

    # ---------------------------------------------------------
    # Chunk document
    # ---------------------------------------------------------

    chunker = MarkdownChunker(
        max_chars=2000
    )

    chunks = chunker.chunk(
        content=document["content"],
        metadata=document["metadata"],
    )

    print(f"\nDocument: {FILE_PATH}")
    print(f"Chunks: {len(chunks)}")

    # ---------------------------------------------------------
    # Create embedder
    # ---------------------------------------------------------

    embedder = OllamaEmbedder(
        model="nomic-embed-text"
    )

    # ---------------------------------------------------------
    # Embed first chunk
    # ---------------------------------------------------------

    chunk = chunks[0]

    print("\n" + "-" * 80)
    print("TEXT BEING EMBEDDED")
    print("-" * 80)

    print(chunk.content)

    print("\nGenerating embedding...")

    embedding = embedder.embed(
        chunk.content
    )

    # ---------------------------------------------------------
    # Display result
    # ---------------------------------------------------------

    print("\n" + "-" * 80)
    print("EMBEDDING RESULT")
    print("-" * 80)

    print(f"Model: {embedder.model}")
    print(f"Dimensions: {len(embedding)}")

    print("\nFirst 10 values:")

    for value in embedding[:10]:
        print(f"  {value}")

    print("\nEmbedding test completed successfully.")


if __name__ == "__main__":
    main()