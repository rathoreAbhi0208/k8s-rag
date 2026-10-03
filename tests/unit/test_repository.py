from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.chunker import DocumentChunk
from src.ingestion.hasher import generate_chunk_id
from src.vectorstore.repository import MilvusRepository


COLLECTION_NAME = "test_collection"
DIMENSION = 3


@pytest.fixture
def repository():
    with patch(
        "src.vectorstore.repository.LocalMilvusClient"
    ) as mock_client_class:

        mock_client = MagicMock()

        mock_client_class.return_value = mock_client

        repository = MilvusRepository(
            collection_name=COLLECTION_NAME,
            dimension=DIMENSION,
            db_path="test.db",
        )

        repository._mock_client = mock_client

        yield repository


def test_repository_creates_collection(repository):
    repository._mock_client.create_kubernetes_collection.assert_called_once_with(
        collection_name=COLLECTION_NAME,
        dimension=DIMENSION,
    )


def test_insert_chunks(repository):
    chunks = [
        DocumentChunk(
            content="Kubernetes control plane",
            metadata={
                "source": "components.md",
                "filename": "components.md",
                "heading": "Control Plane",
                "chunk_index": 0,
            },
        ),
        DocumentChunk(
            content="Kubernetes worker nodes",
            metadata={
                "source": "components.md",
                "filename": "components.md",
                "heading": "Node Components",
                "chunk_index": 1,
            },
        ),
    ]

    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    repository._mock_client.client.insert.return_value = {
        "ids": [
            generate_chunk_id(
                "components.md",
                0,
            ),
            generate_chunk_id(
                "components.md",
                1,
            ),
        ]
    }

    result = repository.insert_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    expected_id_0 = generate_chunk_id(
        "components.md",
        0,
    )

    expected_id_1 = generate_chunk_id(
        "components.md",
        1,
    )

    assert result == [
        expected_id_0,
        expected_id_1,
    ]

    repository._mock_client.client.insert.assert_called_once()

    call_args = (
        repository._mock_client.client.insert.call_args
    )

    inserted_data = call_args.kwargs["data"]

    assert len(inserted_data) == 2

    assert inserted_data[0]["id"] == expected_id_0
    assert inserted_data[0]["content"] == (
        "Kubernetes control plane"
    )
    assert inserted_data[0]["source"] == (
        "components.md"
    )
    assert inserted_data[0]["filename"] == (
        "components.md"
    )
    assert inserted_data[0]["heading"] == (
        "Control Plane"
    )
    assert inserted_data[0]["chunk_index"] == 0
    assert inserted_data[0]["embedding"] == [
        0.1,
        0.2,
        0.3,
    ]

    assert inserted_data[1]["id"] == expected_id_1
    assert inserted_data[1]["content"] == (
        "Kubernetes worker nodes"
    )
    assert inserted_data[1]["source"] == (
        "components.md"
    )
    assert inserted_data[1]["filename"] == (
        "components.md"
    )
    assert inserted_data[1]["heading"] == (
        "Node Components"
    )
    assert inserted_data[1]["chunk_index"] == 1
    assert inserted_data[1]["embedding"] == [
        0.4,
        0.5,
        0.6,
    ]


def test_insert_chunks_uses_deterministic_ids(
    repository,
):
    chunks = [
        DocumentChunk(
            content="Kubernetes",
            metadata={
                "source": "test.md",
                "filename": "test.md",
                "heading": "Test",
                "chunk_index": 5,
            },
        ),
    ]

    embeddings = [
        [0.1, 0.2, 0.3],
    ]

    expected_id = generate_chunk_id(
        "test.md",
        5,
    )

    repository._mock_client.client.insert.return_value = {
        "ids": [expected_id]
    }

    repository.insert_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    inserted_data = (
        repository._mock_client.client.insert.call_args
        .kwargs["data"]
    )

    assert inserted_data[0]["id"] == expected_id


def test_insert_chunks_rejects_mismatched_lengths(
    repository,
):
    chunks = [
        DocumentChunk(
            content="Kubernetes",
            metadata={},
        ),
    ]

    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    with pytest.raises(
        ValueError,
        match="Number of chunks must match number of embeddings",
    ):
        repository.insert_chunks(
            chunks=chunks,
            embeddings=embeddings,
        )

    repository._mock_client.client.insert.assert_not_called()


def test_insert_empty_chunks(repository):
    result = repository.insert_chunks(
        chunks=[],
        embeddings=[],
    )

    assert result == []

    repository._mock_client.client.insert.assert_not_called()


def test_count(repository):
    repository._mock_client.client.query.return_value = [
        {"id": 1},
        {"id": 2},
        {"id": 3},
    ]

    result = repository.count()

    assert result == 3

    repository._mock_client.client.query.assert_called_once_with(
        collection_name=COLLECTION_NAME,
        filter="",
        output_fields=["id"],
        limit=10000,
    )

# def test_get_all_chunks(repository, mock_client):
#     mock_client.query.return_value = [
#         {
#             "id": 1,
#             "content": "Kubernetes control plane",
#             "source": "components.md",
#             "filename": "components.md",
#             "heading": "Control Plane Components",
#             "chunk_index": 0,
#         },
#         {
#             "id": 2,
#             "content": "Kubernetes worker nodes",
#             "source": "components.md",
#             "filename": "components.md",
#             "heading": "Node Components",
#             "chunk_index": 1,
#         },
#     ]

#     result = repository.get_all_chunks()

#     assert len(result) == 2
#     assert result[0]["heading"] == "Control Plane Components"

#     mock_client.query.assert_called_once_with(
#         collection_name=repository.collection_name,
#         filter="",
#         output_fields=[
#             "id",
#             "content",
#             "source",
#             "filename",
#             "heading",
#             "chunk_index",
#         ],
#         limit=10000,
#     )