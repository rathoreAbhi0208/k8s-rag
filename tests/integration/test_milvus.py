from pathlib import Path

from pymilvus import MilvusClient

from src.vectorstore.collections import (
    EMBEDDING_DIMENSION,
)
from src.vectorstore.milvus_client import (
    LocalMilvusClient,
)


COLLECTION_NAME = "test_kubernetes_docs"


def create_test_collection(
    client: LocalMilvusClient,
):
    """Create a temporary test collection."""

    client.create_kubernetes_collection(
        collection_name=COLLECTION_NAME,
        dimension=EMBEDDING_DIMENSION,
    )


def test_create_milvus_collection(
    tmp_path: Path,
):
    """Test creating a Milvus collection."""

    db_path = tmp_path / "test_milvus.db"

    client = LocalMilvusClient(
        db_path=str(db_path),
    )

    create_test_collection(client)

    assert client.has_collection(
        COLLECTION_NAME
    )

    assert COLLECTION_NAME in (
        client.list_collections()
    )


def test_insert_and_query_vector(
    tmp_path: Path,
):
    """Test inserting and querying a vector."""

    db_path = tmp_path / "test_milvus.db"

    client = LocalMilvusClient(
        db_path=str(db_path),
    )

    create_test_collection(client)

    vector = [0.1] * EMBEDDING_DIMENSION

    data = [
        {
            "id": 1,
            "content": "Kubernetes control plane manages the cluster.",
            "source": "test/components.md",
            "filename": "components.md",
            "heading": "Control Plane Components",
            "chunk_index": 0,
            "embedding": vector,
        }
    ]

    result = client.client.insert(
        collection_name=COLLECTION_NAME,
        data=data,
    )

    assert len(result["ids"]) == 1
    assert result["ids"][0] == 1

    query_result = client.client.get(
        collection_name=COLLECTION_NAME,
        ids=[1],
        output_fields=[
            "id",
            "content",
            "source",
            "filename",
            "heading",
            "chunk_index",
        ],
    )

    assert len(query_result) == 1

    record = query_result[0]

    assert record["id"] == 1
    assert (
        record["content"]
        == "Kubernetes control plane manages the cluster."
    )
    assert record["source"] == "test/components.md"
    assert record["filename"] == "components.md"
    assert record["heading"] == "Control Plane Components"
    assert record["chunk_index"] == 0


def test_vector_search(
    tmp_path: Path,
):
    """Test semantic vector search."""

    db_path = tmp_path / "test_milvus.db"

    client = LocalMilvusClient(
        db_path=str(db_path),
    )

    create_test_collection(client)

    vector_1 = [0.1] * EMBEDDING_DIMENSION
    vector_2 = [0.9] * EMBEDDING_DIMENSION

    data = [
        {
            "id": 1,
            "content": "Kubernetes control plane.",
            "source": "control-plane.md",
            "filename": "control-plane.md",
            "heading": "Control Plane",
            "chunk_index": 0,
            "embedding": vector_1,
        },
        {
            "id": 2,
            "content": "Kubernetes worker nodes.",
            "source": "nodes.md",
            "filename": "nodes.md",
            "heading": "Node Components",
            "chunk_index": 0,
            "embedding": vector_2,
        },
    ]

    client.client.insert(
        collection_name=COLLECTION_NAME,
        data=data,
    )

    query_vector = [0.1] * EMBEDDING_DIMENSION

    results = client.client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        limit=2,
        output_fields=[
            "content",
            "heading",
        ],
    )

    assert len(results) == 1
    assert len(results[0]) == 2

    first_result = results[0][0]

    assert first_result["id"] == 1
    assert (
        first_result["entity"]["content"]
        == "Kubernetes control plane."
    )
    assert (
        first_result["entity"]["heading"]
        == "Control Plane"
    )


def test_drop_collection(
    tmp_path: Path,
):
    """Test deleting a collection."""

    db_path = tmp_path / "test_milvus.db"

    client = LocalMilvusClient(
        db_path=str(db_path),
    )

    create_test_collection(client)

    assert client.has_collection(
        COLLECTION_NAME
    )

    client.drop_collection(
        COLLECTION_NAME
    )

    assert not client.has_collection(
        COLLECTION_NAME
    )