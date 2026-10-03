from pathlib import Path
from typing import Any

from pymilvus import MilvusClient


MILVUS_DB = Path("data/milvus.db")
COLLECTION_NAME = "kubernetes_docs"


def get_client() -> MilvusClient:
    """Create and return a local Milvus client."""
    if not MILVUS_DB.exists():
        raise FileNotFoundError(
            f"Milvus database not found: {MILVUS_DB}"
        )

    return MilvusClient(str(MILVUS_DB))


def list_collections(client: MilvusClient) -> list[str]:
    """List all collections in the local Milvus database."""
    collections = client.list_collections()

    print("\n" + "=" * 80)
    print("COLLECTIONS")
    print("=" * 80)

    if not collections:
        print("No collections found.")
        return []

    for collection in collections:
        print(f"- {collection}")

    return collections


def ensure_collection_exists(
    client: MilvusClient,
    collection_name: str,
) -> None:
    """Raise an error if the requested collection does not exist."""
    if not client.has_collection(collection_name):
        raise ValueError(
            f"Collection does not exist: {collection_name}"
        )


def ensure_collection_loaded(
    client: MilvusClient,
    collection_name: str,
) -> None:
    """Load a collection before query/search operations."""
    ensure_collection_exists(
        client=client,
        collection_name=collection_name,
    )

    client.load_collection(
        collection_name=collection_name,
    )


def show_schema(
    client: MilvusClient,
    collection_name: str,
) -> dict[str, Any]:
    """Display and return the collection schema."""
    ensure_collection_exists(
        client=client,
        collection_name=collection_name,
    )

    collection = client.describe_collection(
        collection_name=collection_name,
    )

    print("\n" + "=" * 80)
    print(f"SCHEMA: {collection_name}")
    print("=" * 80)

    schema = collection.get("schema", collection)

    print(schema)

    return collection


def get_count(
    client: MilvusClient,
    collection_name: str,
) -> int:
    """Return the number of records in a collection."""
    ensure_collection_loaded(
        client=client,
        collection_name=collection_name,
    )

    result = client.query(
        collection_name=collection_name,
        filter="",
        output_fields=["id"],
        limit=16384,
    )

    count = len(result)

    print("\n" + "=" * 80)
    print(f"DOCUMENT COUNT: {collection_name}")
    print("=" * 80)
    print(f"Count: {count}")

    return count


def get_sample_data(
    client: MilvusClient,
    collection_name: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return and display sample records."""
    if limit <= 0:
        raise ValueError("limit must be greater than 0")

    ensure_collection_loaded(
        client=client,
        collection_name=collection_name,
    )

    results = client.query(
        collection_name=collection_name,
        filter="",
        output_fields=[
            "id",
            "content",
            "source",
            "filename",
            "heading",
            "chunk_index",
            "heading_level",
            "heading_path",
        ],
        limit=limit,
    )

    print("\n" + "=" * 80)
    print(f"SAMPLE DATA: {collection_name}")
    print("=" * 80)

    if not results:
        print("No data found.")
        return []

    for index, row in enumerate(results, start=1):
        print(f"\n--- Sample {index} ---")
        print(f"ID:             {row.get('id')}")
        print(f"Source:         {row.get('source')}")
        print(f"Filename:       {row.get('filename')}")
        print(f"Heading:        {row.get('heading')}")
        print(f"Heading Level:  {row.get('heading_level')}")
        print(f"Heading Path:   {row.get('heading_path')}")
        print(f"Chunk Index:    {row.get('chunk_index')}")
        print(f"Content:        {row.get('content')}")

    return results


def inspect_collection(
    client: MilvusClient,
    collection_name: str,
    sample_limit: int = 5,
) -> None:
    """Inspect schema, count, and sample records."""
    ensure_collection_exists(
        client=client,
        collection_name=collection_name,
    )

    show_schema(
        client=client,
        collection_name=collection_name,
    )

    get_count(
        client=client,
        collection_name=collection_name,
    )

    get_sample_data(
        client=client,
        collection_name=collection_name,
        limit=sample_limit,
    )


def main() -> None:
    """Inspect the local Kubernetes Milvus collection."""
    print("\n" + "=" * 80)
    print("MILVUS LOCAL DATABASE INSPECTION")
    print("=" * 80)

    print(f"Database:    {MILVUS_DB}")
    print(f"Collection:  {COLLECTION_NAME}")

    client = get_client()

    collections = list_collections(client)

    if COLLECTION_NAME not in collections:
        print(
            f"\nCollection '{COLLECTION_NAME}' was not found."
        )
        return

    inspect_collection(
        client=client,
        collection_name=COLLECTION_NAME,
        sample_limit=5,
    )


if __name__ == "__main__":
    main()