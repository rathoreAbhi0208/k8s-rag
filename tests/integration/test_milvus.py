import uuid

from src.vectorstore.milvus_client import LocalMilvusClient


DIMENSION = 3


def create_test_collection():
    collection_name = f"test_kubernetes_docs_{uuid.uuid4().hex[:8]}"

    milvus = LocalMilvusClient(
        db_path="data/test_milvus.db",
    )

    milvus.create_kubernetes_collection(
        collection_name=collection_name,
        dimension=DIMENSION,
    )

    return milvus, collection_name


def test_insert_and_query_vector():
    milvus, collection_name = create_test_collection()

    try:
        rows = [
            {
                "id": 1,
                "content": "Kubernetes control plane manages the cluster.",
                "source": "components.md",
                "filename": "components.md",
                "heading": "Control Plane Components",
                "chunk_index": 0,
                "heading_level": 2,
                "heading_path": "Core Components > Control Plane Components",
                "embedding": [0.1, 0.2, 0.3],
            }
        ]

        result = milvus.client.insert(
            collection_name=collection_name,
            data=rows,
        )

        assert result["insert_count"] == 1

        query_result = milvus.client.query(
            collection_name=collection_name,
            filter="id == 1",
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
        )

        assert len(query_result) == 1

        row = query_result[0]

        assert row["id"] == 1
        assert row["content"] == (
            "Kubernetes control plane manages the cluster."
        )
        assert row["source"] == "components.md"
        assert row["filename"] == "components.md"
        assert row["heading"] == "Control Plane Components"
        assert row["chunk_index"] == 0
        assert row["heading_level"] == 2
        assert row["heading_path"] == (
            "Core Components > Control Plane Components"
        )

    finally:
        milvus.drop_collection(collection_name)


def test_vector_search():
    milvus, collection_name = create_test_collection()

    try:
        rows = [
            {
                "id": 1,
                "content": "Kubernetes control plane manages the cluster.",
                "source": "components.md",
                "filename": "components.md",
                "heading": "Control Plane Components",
                "chunk_index": 0,
                "heading_level": 2,
                "heading_path": "Core Components > Control Plane Components",
                "embedding": [0.1, 0.2, 0.3],
            },
            {
                "id": 2,
                "content": "Worker nodes run application workloads.",
                "source": "components.md",
                "filename": "components.md",
                "heading": "Node Components",
                "chunk_index": 1,
                "heading_level": 2,
                "heading_path": "Core Components > Node Components",
                "embedding": [0.9, 0.8, 0.7],
            },
        ]

        insert_result = milvus.client.insert(
            collection_name=collection_name,
            data=rows,
        )

        assert insert_result["insert_count"] == 2

        results = milvus.client.search(
            collection_name=collection_name,
            data=[[0.1, 0.2, 0.3]],
            anns_field="embedding",
            limit=2,
            output_fields=[
                "content",
                "source",
                "filename",
                "heading",
                "chunk_index",
                "heading_level",
                "heading_path",
            ],
        )

        assert len(results) == 1
        assert len(results[0]) == 2

        first = results[0][0]

        assert first["id"] == 1
        assert first["entity"]["content"] == (
            "Kubernetes control plane manages the cluster."
        )
        assert first["entity"]["heading"] == "Control Plane Components"
        assert first["entity"]["chunk_index"] == 0
        assert first["entity"]["heading_level"] == 2
        assert first["entity"]["heading_path"] == (
            "Core Components > Control Plane Components"
        )

    finally:
        milvus.drop_collection(collection_name)