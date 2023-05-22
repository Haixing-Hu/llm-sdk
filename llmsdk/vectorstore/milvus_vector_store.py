# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import Dict, Optional, Any, List
import uuid

import pymilvus

from ..common import Vector, Point
from ..criterion import Criterion
from .distance import Distance
from .payload_schema import PayloadSchema
from .collection_info import CollectionInfo
from .vector_store import VectorStore
from .milvus_utils import to_milvus_field_schema, criterion_to_expr, to_milvus_distance


class MilvusVectorStore(VectorStore):
    """
    The vector store based on the Milvus vector database.
    """

    DEFAULT_ID_FIELD_NAME = "__id__"
    """
    The default field name of the primary ID field.
    """

    DEFAULT_VECTOR_FIELD_NAME = "__vector__"
    """
    The default field name of the floating vector field.
    """

    DEFAULT_VECTOR_INDEX_TYPE = "HNSW"
    """
    The default vector index type.
    """

    DEFAULT_INDEX_PARAMS = {
        "IVF_FLAT": {"nprobe": 10},
        "IVF_SQ8": {"nprobe": 10},
        "IVF_PQ": {"nprobe": 10},
        "HNSW": {"ef": 10},
        "RHNSW_FLAT": {"ef": 10},
        "RHNSW_SQ": {"ef": 10},
        "RHNSW_PQ": {"ef": 10},
        "IVF_HNSW": {"nprobe": 10, "ef": 10},
        "ANNOY": {"search_k": 10},
    }
    """
    The default index params for different index types.
    """

    def __init__(self,
                 connection_args: Optional[Dict] = None) -> None:
        """
        Construct a vector store based on a collection of a Milvus vector
        database.

        :param connection_args: the arguments for the database connection.
        """
        super().__init__()
        if connection_args is None:
            self._connection_args = {}
            self._connection_alias = "default"
        else:
            self._connection_args = connection_args
            self._connection_alias = connection_args.get("alias", "default")
        self._collection = None
        self._auto_id = None
        self._fields = None
        self._id_field = None
        self._vector_field = None
        self._metadata_fields = None
        self._vector_index = None

    def open(self) -> None:
        # Connecting to Milvus instance
        if not pymilvus.connections.has_connection(self._connection_alias):
            pymilvus.connections.connect(**self._connection_args)

    def close(self) -> None:
        if self._collection is not None:
            self._collection.release()
        pymilvus.connections.disconnect(self._connection_alias)

    def open_collection(self,
                        collection_name: str,
                        id_field: Optional[str] = None,
                        vector_field: Optional[str] = None) -> None:
        """
        Opens the specified collection, and sets it as the current collection.

        :param collection_name: the name of the collection in the vector database.
        :param id_field: the name of ID field in the collection.
        :param vector_field: the name of vector field in the collection.
        """
        super().open_collection(collection_name)
        self._collection = pymilvus.Collection(name=collection_name,
                                               using=self._connection_alias)
        self._auto_id = self._collection.schema.auto_id
        self._id_field = id_field
        self._vector_field = vector_field
        self._fields = []
        # grab the fields for the existing collection.
        for f in self._collection.schema.fields:
            self._fields.append(f.name)
            if f.is_primary and self._id_field is None:
                self._id_field = f.name
            elif f.dtype == pymilvus.DataType.FLOAT_VECTOR and self._vector_field is None:
                self._vector_field = f.name
        if self._id_field is None:
            raise ValueError(f"No primary field in the collection '{collection_name}'.")
        if self._vector_field is None:
            raise ValueError(f"No float vector field in the collection '{collection_name}'.")
        if self._id_field not in self._fields:
            raise ValueError(f"The ID field '{self._id_field}' is not in the "
                             f"collection '{collection_name}'.")
        if self._vector_field not in self._fields:
            raise ValueError(f"The vector field '{self._vector_field}' is not in "
                             f"the collection '{collection_name}'.")
        # find the index for the vector field
        self._vector_index = None
        for index in self._collection.indexes:
            if index.field_name == self._vector_field:
                self._vector_index = index
        if self._vector_index is None:
            raise ValueError(f"No index for the vector field '{self._vector_field}' "
                             f"in the collection '{collection_name}'.")
        # exclude the ID field and vector field from the metadata fields
        self._metadata_fields = self._fields.copy()
        self._metadata_fields.remove(self._id_field)
        self._metadata_fields.remove(self._vector_field)
        # load the collection
        self._collection.load()

    def close_collection(self) -> None:
        super().close_collection()
        if self._collection is not None:
            self._collection.release()
            self._collection = None

    def create_collection(self,
                          collection_name: str,
                          vector_size: int,
                          distance: Distance = Distance.COSINE,
                          payload_schemas: List[PayloadSchema] = None) -> None:
        # prepare the collection schema
        id_field_name = MilvusVectorStore.DEFAULT_ID_FIELD_NAME
        vector_field_name = MilvusVectorStore.DEFAULT_VECTOR_FIELD_NAME
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
        # Create the index for the vector field
        vector_metric_type = to_milvus_distance(distance)
        vector_index_type = MilvusVectorStore.DEFAULT_VECTOR_INDEX_TYPE
        vector_index_params = MilvusVectorStore.DEFAULT_INDEX_PARAMS[vector_index_type]
        collection.create_index(field_name=vector_field_name,
                                index_params={
                                    "metric_type": vector_metric_type,
                                    "index_type": vector_index_type,
                                    "params": vector_index_params,
                                })

        pass

    def delete_collection(self, collection_name: str) -> None:
        pymilvus.utility.drop_collection(collection_name)

    def get_collection_info(self, collection_name: str) -> CollectionInfo:
        pass

    def add(self,
            point: Point,
            **kwargs: Any) -> str:
        ids = self.add_all([point], **kwargs)
        return ids[0]

    def add_all(self,
                points: List[Point],
                **kwargs: Any) -> List[str]:
        data: List[List[Any]] = [] * len(self._fields)
        for p in points:
            if not self._auto_id and p.id is None:
                p.id = str(uuid.uuid4())
            for i, field in enumerate(self._fields):
                if field == self._id_field:
                    data[i].append(p.id)
                elif field == self._vector_field:
                    data[i].append(p.vector)
                else:
                    if p.metadata is None or field not in p.metadata.keys():
                        value = None
                    else:
                        value = p.metadata[field]
                    data[i].append(value)
        self._logger.debug("Insert data: %s", data)
        result = self._collection.insert(data=data, **kwargs)
        self._collection.flush()
        # set the automatically generated IDs for points
        if self._auto_id:
            for i, value in enumerate(result.primary_keys):
                points[i].id = value
        return [p.id for p in points]

    def search(self,
               vector: Vector,
               limit: int,
               criterion: Optional[Criterion] = None,
               **kwargs: Any) -> List[Point]:
        params = {"metric_type": self._vector_index.params["metric_type"]}
        index_type = self._vector_index.params["index_type"]
        if index_type in DEFAULT_INDEX_PARAMS:
            params["params"] = DEFAULT_INDEX_PARAMS[index_type]
        else:
            params["params"] = self._vector_index.params["params"]
        expr = criterion_to_expr(criterion)
        results = self._collection.search(data=[vector],
                                          anns_field=self._vector_field,
                                          param=params,
                                          limit=limit,
                                          expr=expr,
                                          output_fields=self._fields,
                                          **kwargs)
        points = []
        for r in results[0]:
            vector = r.entity.get(self._vector_field)
            metadata = {}
            for f in self._metadata_fields:
                v = r.entity.get(f)
                if v is not None:
                    metadata[f] = v
            point = Point(id=r.id,
                          vector=vector,
                          metadata=metadata,
                          score=r.distance)
            points.append(point)
        return points
