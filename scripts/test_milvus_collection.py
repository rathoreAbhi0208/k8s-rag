from src.vectorstore.collections import (
    EMBEDDING_DIMENSION,
    KUBERNETES_COLLECTION,
)
from src.vectorstore.milvus_client import (
    LocalMilvusClient,
)


def main():
    print("\n" + "=" * 80)
    print("MILVUS COLLECTION TEST")
    print("=" * 80)

    milvus = LocalMilvusClient()

    print("\nCreating collection:")
    print(f"  Name: {KUBERNETES_COLLECTION}")
    print(f"  Dimension: {EMBEDDING_DIMENSION}")
    print("  Metric: COSINE")

    milvus.create_collection(
        collection_name=KUBERNETES_COLLECTION,
        dimension=EMBEDDING_DIMENSION,
    )

    print("\nCollections:")

    for collection in milvus.list_collections():
        print(f"  - {collection}")

    print("\nCollection test completed successfully.")


if __name__ == "__main__":
    main()