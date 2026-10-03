from pathlib import Path
from unittest.mock import Mock

from src.ingestion.pipeline import IngestionPipeline


def test_ingest_file():
    repository = Mock()
    embedder = Mock()
    chunker = Mock()
    loader = Mock()

    file_path = Path("test.md")

    loader.load.return_value = {
        "content": "Control plane\n\nWorker nodes",
        "metadata": {
            "source": str(file_path),
            "filename": "test.md",
            "file_type": ".md",
            "file_size": 100,
            "title": "Kubernetes Components",
        },
    }

    chunker.chunk.return_value = [
        Mock(
            content="Control plane",
            metadata={
                "source": str(file_path),
                "filename": "test.md",
                "title": "Kubernetes Components",
                "heading": "Control Plane",
                "heading_level": 2,
                "heading_path": "Core Components > Control Plane",
                "chunk_index": 0,
            },
        ),
        Mock(
            content="Worker nodes",
            metadata={
                "source": str(file_path),
                "filename": "test.md",
                "title": "Kubernetes Components",
                "heading": "Node Components",
                "heading_level": 2,
                "heading_path": "Core Components > Node Components",
                "chunk_index": 1,
            },
        ),
    ]

    embedder.embed_batch.return_value = [
        [0.1, 0.2],
        [0.3, 0.4],
    ]

    repository.insert_chunks.return_value = [
        111,
        222,
    ]

    pipeline = IngestionPipeline(
        repository=repository,
        embedder=embedder,
        chunker=chunker,
        loader=loader,
    )

    result = pipeline.ingest_file(file_path)

    loader.load.assert_called_once_with(file_path)

    chunker.chunk.assert_called_once_with(
        content="Control plane\n\nWorker nodes",
        metadata={
            "source": str(file_path),
            "filename": "test.md",
            "file_type": ".md",
            "file_size": 100,
            "title": "Kubernetes Components",
        },
    )

    embedder.embed_batch.assert_called_once_with(
        [
            (
                "Document: Kubernetes Components\n"
                "Section: Core Components > Control Plane\n\n"
                "Control plane"
            ),
            (
                "Document: Kubernetes Components\n"
                "Section: Core Components > Node Components\n\n"
                "Worker nodes"
            ),
        ]
    )

    repository.insert_chunks.assert_called_once_with(
        chunks=chunker.chunk.return_value,
        embeddings=[
            [0.1, 0.2],
            [0.3, 0.4],
        ],
    )

    assert result == {
        "source": str(file_path),
        "chunks": 2,
        "embeddings": 2,
        "inserted_ids": [111, 222],
    }


def test_ingest_empty_document():
    repository = Mock()
    embedder = Mock()
    chunker = Mock()
    loader = Mock()

    file_path = Path("empty.md")

    loader.load.return_value = {
        "content": "",
        "metadata": {
            "source": str(file_path),
            "filename": "empty.md",
        },
    }

    chunker.chunk.return_value = []

    pipeline = IngestionPipeline(
        repository=repository,
        embedder=embedder,
        chunker=chunker,
        loader=loader,
    )

    result = pipeline.ingest_file(file_path)

    assert result == {
        "source": str(file_path),
        "chunks": 0,
        "embeddings": 0,
        "inserted_ids": [],
    }

    embedder.embed_batch.assert_not_called()
    repository.insert_chunks.assert_not_called()