from pathlib import Path

from src.vectorstore.milvus_client import (
    LocalMilvusClient,
)


def test_milvus_client_creates_database(
    tmp_path: Path,
):
    db_path = tmp_path / "test.db"

    client = LocalMilvusClient(
        db_path=str(db_path),
    )

    assert client.client is not None
    assert db_path.parent.exists()


def test_collection_does_not_exist_initially(
    tmp_path: Path,
):
    db_path = tmp_path / "test.db"

    client = LocalMilvusClient(
        db_path=str(db_path),
    )

    assert not client.has_collection(
        "test_collection"
    )


def test_collection_list_is_empty_initially(
    tmp_path: Path,
):
    db_path = tmp_path / "test.db"

    client = LocalMilvusClient(
        db_path=str(db_path),
    )

    assert client.list_collections() == []