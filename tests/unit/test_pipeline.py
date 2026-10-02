from pathlib import Path
from unittest.mock import MagicMock

from src.ingestion.chunker import DocumentChunk
from src.ingestion.pipeline import IngestionPipeline


def test_ingest_file():
    loader = MagicMock()
    chunker = MagicMock()
    embedder = MagicMock()
    repository = MagicMock()

    loader.load.return_value = {
        "content": "Kubernetes content",
        "metadata": {
            "source": "components.md",
            "filename": "components.md",
        },
    }

    chunks = [
        DocumentChunk(
            content="Control plane",
            metadata={
                "source": "components.md",
                "filename": "components.md",
                "heading": "Control Plane",
                "chunk_index": 0,
            },
        ),
        DocumentChunk(
            content="Worker nodes",
            metadata={
                "source": "components.md",
                "filename": "components.md",
                "heading": "Node Components",
                "chunk_index": 1,
            },
        ),
    ]

    chunker.chunk.return_value = chunks

    embedder.embed_batch.return_value = [
        [0.1, 0.2],
        [0.3, 0.4],
    ]

    repository.insert_chunks.return_value = [
        100,
        200,
    ]

    pipeline = IngestionPipeline(
        repository=repository,
        embedder=embedder,
        chunker=chunker,
        loader=loader,
    )

    result = pipeline.ingest_file(
        "components.md"
    )

    loader.load.assert_called_once_with(
        Path("components.md")
    )

    chunker.chunk.assert_called_once_with(
        content="Kubernetes content",
        metadata={
            "source": "components.md",
            "filename": "components.md",
        },
    )

    embedder.embed_batch.assert_called_once_with(
        [
            "Heading: Control Plane\n\nControl plane",
            "Heading: Node Components\n\nWorker nodes",
        ]
    )

    repository.insert_chunks.assert_called_once_with(
        chunks=chunks,
        embeddings=[
            [0.1, 0.2],
            [0.3, 0.4],
        ],
    )

    assert result["source"] == "components.md"
    assert result["chunks"] == 2
    assert result["embeddings"] == 2
    assert result["inserted_ids"] == [
        100,
        200,
    ]


def test_ingest_file_with_no_chunks():
    loader = MagicMock()
    chunker = MagicMock()
    embedder = MagicMock()
    repository = MagicMock()

    loader.load.return_value = {
        "content": "",
        "metadata": {
            "source": "empty.md",
        },
    }

    chunker.chunk.return_value = []

    pipeline = IngestionPipeline(
        repository=repository,
        embedder=embedder,
        chunker=chunker,
        loader=loader,
    )

    result = pipeline.ingest_file(
        "empty.md"
    )

    assert result == {
        "source": "empty.md",
        "chunks": 0,
        "embeddings": 0,
        "inserted_ids": [],
    }

    embedder.embed_batch.assert_not_called()
    repository.insert_chunks.assert_not_called()