# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import Dict, Optional, Any, List

import pymilvus

from .distance import Distance
from .payload_schema import PayloadSchema
from .collection_info import CollectionInfo
from .vector_store import VectorStore
from .milvus_utils import (
    to_milvus_field_schema,
    criterion_to_expr,
    to_milvus_distance,
    get_id_field,
    get_vector_field,
    get_index,
    to_local_distance,
    get_payload_schemas,
    DEFAULT_ID_FIELD_NAME,
    DEFAULT_VECTOR_FIELD_NAME,
    DEFAULT_VECTOR_INDEX_TYPE,
    DEFAULT_INDEX_PARAMS,
)
from ..common import Vector, Point, Metadata
from ..criterion import Criterion
from ..generator import IdGenerator


class MilvusVectorStore(VectorStore):
    """
    The vector store based on the Milvus vector database.
    """

    def __init__(self,
                 connection_args: Optional[Dict] = None,
                 id_generator: Optional[IdGenerator] = None) -> None:
        """
        Construct a vector store based on a collection of a Milvus vector
        database.

        :param connection_args: the arguments for the database connection.
        :param id_generator: the ID generator used to generate ID of documents.
        """
        super().__init__(id_generator=id_generator)
        if connection_args is None:
            self._connection_args = {}
            self._connection_alias = "default"
        else:
            self._connection_args = connection_args
            self._connection_alias = connection_args.get("alias", "default")
        self._collection: Optional[pymilvus.Collection] = None
        self._auto_id: Optional[bool] = None
        self._id_field: Optional[pymilvus.FieldSchema] = None
        self._vector_field: Optional[pymilvus.FieldSchema] = None
        self._vector_index: Optional[pymilvus.Index] = None
        self._payload_schemas: Optional[List[PayloadSchema]] = None

    def open(self) -> None:
        # Connecting to Milvus instance
        if not pymilvus.connections.has_connection(self._connection_alias):
            pymilvus.connections.connect(**self._connection_args)

    def close(self) -> None:
        if self._collection is not None:
            self.close_collection()
        pymilvus.connections.disconnect(self._connection_alias)

    def open_collection(self,
                        collection_name: str,
                        id_field_name: Optional[str] = None,
                        vector_field_name: Optional[str] = None) -> None:
        """
        Opens the specified collection, and sets it as the current collection.

        :param collection_name: the name of the collection in the vector database.
        :param id_field_name: the name of ID field in the collection.
        :param vector_field_name: the name of vector field in the collection.
        """
        super().open_collection(collection_name)
        self._collection = pymilvus.Collection(name=collection_name,
                                               using=self._connection_alias)
        self._auto_id = self._collection.schema.auto_id
        self._id_field = get_id_field(self._collection, id_field_name)
        self._vector_field = get_vector_field(self._collection, vector_field_name)
        self._vector_index = get_index(self._collection, self._vector_field.name)
        self._payload_schemas = get_payload_schemas(self._collection,
                                                    id_field=self._id_field,
                                                    vector_field=self._vector_field)
        self._collection.load()

    def close_collection(self) -> None:
        super().close_collection()
        if self._collection is not None:
            self._collection.release()
            self._collection = None
            self._auto_id = None
            self._id_field = None
            self._vector_field = None
            self._vector_index = None
            self._payload_schemas = None

    def create_collection(self,
                          collection_name: str,
                          vector_size: int,
                          distance: Distance = Distance.COSINE,
                          payload_schemas: List[PayloadSchema] = None) -> None:
        # prepare the collection schema
        id_field_name = DEFAULT_ID_FIELD_NAME
        vector_field_name = DEFAULT_VECTOR_FIELD_NAME
        id_field = pymilvus.FieldSchema(name=id_field_name,
                                        dtype=pymilvus.DataType.STRING,
                                        is_primary=True)
        vector_field = pymilvus.FieldSchema(name=vector_field_name,
                                            dtype=pymilvus.DataType.FLOAT_VECTOR,
                                            dim=vector_size)
        fields = [id_field, vector_field]
        if payload_schemas is not None:
            for schema in payload_schemas:
                fields.append(to_milvus_field_schema(schema))
        collection_schema = pymilvus.CollectionSchema(fields=fields,
                                                      auto_id=False)
        # create the collection
        collection = pymilvus.Collection(name=collection_name,
                                         schema=collection_schema,
                                         using=self._connection_alias)
        # create the index for the vector field
        vector_metric_type = to_milvus_distance(distance)
        vector_index_type = DEFAULT_VECTOR_INDEX_TYPE
        vector_index_params = DEFAULT_INDEX_PARAMS[vector_index_type]
        collection.create_index(field_name=vector_field_name,
                                index_params={
                                    "metric_type": vector_metric_type,
                                    "index_type": vector_index_type,
                                    "params": vector_index_params,
                                })
        # create the index for payload fields
        if payload_schemas is not None:
            for schema in payload_schemas:
                collection.create_index(field_name=schema.name,
                                        index_name=schema.name + "_index")

    def delete_collection(self, collection_name: str) -> None:
        pymilvus.utility.drop_collection(collection_name)

    def get_collection_info(self, collection_name: str) -> CollectionInfo:
        collection = pymilvus.Collection(name=collection_name,
                                         using=self._connection_alias)
        id_field = get_id_field(collection)
        vector_field = get_vector_field(collection)
        vector_index = get_index(collection, vector_field.name)
        vector_index_params = vector_index.params
        distance = to_local_distance(vector_index_params.get("metric_type"))
        payload_schemas = get_payload_schemas(collection, id_field, vector_field)
        return CollectionInfo(name=collection_name,
                              size=collection.num_entities,
                              vector_size=vector_field.params.get("dim"),
                              distance=distance,
                              payload_schemas=payload_schemas)

    def add(self, point: Point) -> str:
        self._ensure_collection_opened()
        ids = self.add_all([point])
        return ids[0]

    def add_all(self, points: List[Point]) -> List[str]:
        self._ensure_collection_opened()
        fields = self._collection.schema.fields
        data: List[List[Any]] = [] * len(fields)
        for p in points:
            if not self._auto_id and p.id is None:
                p.id = self._id_generator.generate()
            for i, field in enumerate(fields):
                if field.name == self._id_field.name:
                    data[i].append(p.id)
                elif field.name == self._vector_field.name:
                    data[i].append(p.vector)
                else:
                    if (p.metadata is None) or (field.name not in p.metadata):
                        value = None
                    else:
                        value = p.metadata[field.name]
                    data[i].append(value)
        self._logger.debug("Insert data: %s", data)
        result = self._collection.insert(data=data)
        self._collection.flush()
        # set the automatically generated IDs for points
        if self._auto_id:
            for i, value in enumerate(result.primary_keys):
                points[i].id = value
        return [p.id for p in points]

    def similarity_search(self,
                          query_vector: Vector,
                          limit: int,
                          criterion: Optional[Criterion] = None,
                          **kwargs: Any) -> List[Point]:
        self._ensure_collection_opened()
        params = {"metric_type": self._vector_index.params["metric_type"]}
        index_type = self._vector_index.params["index_type"]
        if index_type in DEFAULT_INDEX_PARAMS:
            params["params"] = DEFAULT_INDEX_PARAMS[index_type]
        else:
            params["params"] = self._vector_index.params["params"]
        expr = criterion_to_expr(criterion)
        payload_field_names = [f.name for f in self._payload_schemas]
        results = self._collection.search(data=[query_vector],
                                          anns_field=self._vector_field.name,
                                          param=params,
                                          limit=limit,
                                          expr=expr,
                                          output_fields=payload_field_names,
                                          **kwargs)
        points = []
        for r in results[0]:
            # FIXME: can we get the vector field directly?
            query_vector = r.entity.get(self._vector_field.name)
            metadata = Metadata()
            for f in payload_field_names:
                v = r.entity.get(f)
                if v is not None:
                    metadata[f] = v
            point = Point(id=r.id,
                          vector=query_vector,
                          metadata=metadata,
                          score=r.distance)
            points.append(point)
        return points
