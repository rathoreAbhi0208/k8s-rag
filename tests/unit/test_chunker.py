from src.ingestion.chunker import MarkdownChunker

def test_chunk_indexes_are_global():
    content = """
# Kubernetes Components
Kubernetes has several components.
# Control Plane Components
The control plane manages the cluster.
# Node Components
Node components run on every node.
"""
    chunker = MarkdownChunker(
        max_chars=2000
    )

    chunks = chunker.chunk(
        content=content,
        metadata={},
    )

    indexes = [
        chunk.metadata["chunk_index"]
        for chunk in chunks
    ]

    assert indexes == [0, 1, 2]