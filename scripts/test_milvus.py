from pymilvus import MilvusClient


DB_PATH = "data/milvus.db"


def main():
    print("\n" + "=" * 80)
    print("MILVUS LITE TEST")
    print("=" * 80)

    client = MilvusClient(DB_PATH)

    print("\nConnected to:")
    print(DB_PATH)

    collections = client.list_collections()

    print("\nExisting collections:")
    print(collections)

    print("\nMilvus Lite is working successfully.")


if __name__ == "__main__":
    main()