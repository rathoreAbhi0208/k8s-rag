from src.vectorstore.milvus_client import LocalMilvusClient
from src.vectorstore.collections import KUBERNETES_COLLECTION


def main():
    milvus = LocalMilvusClient()

    if milvus.has_collection(KUBERNETES_COLLECTION):
        print(
            f"Dropping collection: "
            f"{KUBERNETES_COLLECTION}"
        )

        milvus.drop_collection(
            KUBERNETES_COLLECTION
        )

    print("Collection reset complete.")


if __name__ == "__main__":
    main()