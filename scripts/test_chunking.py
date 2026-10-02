from pathlib import Path

from src.ingestion.chunker import MarkdownChunker
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
    # ---------------------------------------------------------
    # 1. Check document exists
    # ---------------------------------------------------------

    if not FILE_PATH.exists():
        raise RuntimeError(
            f"File not found: {FILE_PATH}"
        )

    print("\n" + "=" * 80)
    print("KUBERNETES DOCUMENT CHUNKING TEST")
    print("=" * 80)

    print(f"\nProcessing:")
    print(FILE_PATH)

    # ---------------------------------------------------------
    # 2. Load Markdown document
    # ---------------------------------------------------------

    loader = MarkdownLoader()

    document = loader.load(FILE_PATH)

    print("\n" + "-" * 80)
    print("DOCUMENT METADATA")
    print("-" * 80)

    for key, value in document["metadata"].items():
        print(f"{key}: {value}")

    # ---------------------------------------------------------
    # 3. Display original document information
    # ---------------------------------------------------------

    content = document["content"]

    print("\n" + "-" * 80)
    print("DOCUMENT INFORMATION")
    print("-" * 80)

    print(f"Characters: {len(content)}")
    print(f"Lines: {len(content.splitlines())}")

    # ---------------------------------------------------------
    # 4. Chunk document
    # ---------------------------------------------------------

    chunker = MarkdownChunker(
        max_chars=2000
    )

    chunks = chunker.chunk(
        content=content,
        metadata=document["metadata"],
    )

    print("\n" + "-" * 80)
    print("CHUNKING RESULTS")
    print("-" * 80)

    print(f"Total chunks: {len(chunks)}")

    # ---------------------------------------------------------
    # 5. Display every chunk
    # ---------------------------------------------------------

    for index, chunk in enumerate(chunks):

        print("\n" + "=" * 80)
        print(f"CHUNK {index}")
        print("=" * 80)

        print("\nMetadata:")

        for key, value in chunk.metadata.items():
            print(f"  {key}: {value}")

        print("\nContent:")
        print("-" * 80)
        print(chunk.content)

        print("\nChunk statistics:")
        print(f"  Characters: {len(chunk.content)}")
        print(f"  Words: {len(chunk.content.split())}")
        print(f"  Lines: {len(chunk.content.splitlines())}")

    # ---------------------------------------------------------
    # 6. Summary
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"Source file : {FILE_PATH}")
    print(f"Total chunks: {len(chunks)}")

    if chunks:
        average_size = sum(
            len(chunk.content)
            for chunk in chunks
        ) / len(chunks)

        print(
            f"Average chunk size: "
            f"{average_size:.0f} characters"
        )

        print("\nHeadings found:")

        for index, chunk in enumerate(chunks):
            print(
                f"  {index}: "
                f"{chunk.metadata.get('heading', 'N/A')}"
            )

    print("\nChunking test completed successfully.\n")


if __name__ == "__main__":
    main()