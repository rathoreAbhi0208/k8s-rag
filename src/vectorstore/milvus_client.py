from pathlib import Path

from pymilvus import (
    DataType,
    MilvusClient,
)


class LocalMilvusClient:
    """Wrapper around a local Milvus Lite database."""

    def __init__(
        self,
        db_path: str = "data/milvus.db",
    ):
        Path(db_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = MilvusClient(db_path)

    def list_collections(self):
        return self.client.list_collections()

    def has_collection(
        self,
        collection_name: str,
    ) -> bool:
        return self.client.has_collection(
            collection_name
        )

    def create_kubernetes_collection(
        self,
        collection_name: str,
        dimension: int,
    ):
        if self.has_collection(collection_name):
            return

        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )

        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            is_primary=True,
        )

        schema.add_field(
            field_name="content",
            datatype=DataType.VARCHAR,
            max_length=10000,
        )

        schema.add_field(
            field_name="source",
            datatype=DataType.VARCHAR,
            max_length=1000,
        )

        schema.add_field(
            field_name="filename",
            datatype=DataType.VARCHAR,
            max_length=500,
        )

        schema.add_field(
            field_name="heading",
            datatype=DataType.VARCHAR,
            max_length=1000,
        )

        schema.add_field(
            field_name="chunk_index",
            datatype=DataType.INT64,
        )

        schema.add_field(
            field_name="embedding",
            datatype=DataType.FLOAT_VECTOR,
            dim=dimension,
        )

        schema.add_field(
            field_name="heading_level",
            datatype=DataType.INT64,
        )

        schema.add_field(
            field_name="heading_path",
            datatype=DataType.VARCHAR,
            max_length=2000,
        )

        index_params = self.client.prepare_index_params()

        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )

    def drop_collection(
        self,
        collection_name: str,
    ):
        if self.has_collection(collection_name):
            self.client.drop_collection(
                collection_name
            )