# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from typing import Dict, Optional, Any, List

from ..common.distance import Distance
from ..common.metadata import Metadata
from ..common.vector import Vector
from ..common.point import Point
from ..criterion.criterion import Criterion
from ..generator.id_generator import IdGenerator
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
        try:
            import pymilvus
        except ImportError:
            raise ImportError("Milvus is not installed, please install it with "
                              "`pip install pymilvus`.")
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

    def _open(self, **kwargs: Any) -> None:
        # Connecting to Milvus instance
        import pymilvus
        if not pymilvus.connections.has_connection(self._connection_alias):
            pymilvus.connections.connect(**self._connection_args)
        self._is_opened = True

    def _close(self) -> None:
        import pymilvus
        self._close_collection()
        pymilvus.connections.disconnect(self._connection_alias)
        self._is_opened = False

    def _open_collection(self,
                         collection_name: str,
                         id_field_name: Optional[str] = None,
                         vector_field_name: Optional[str] = None) -> None:
        import pymilvus
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
        self._collection_name = collection_name

    def _close_collection(self) -> None:
        if self._collection is not None:
            self._collection.release()
            self._collection = None
            self._auto_id = None
            self._id_field = None
            self._vector_field = None
            self._vector_index = None
            self._payload_schemas = None
        self._collection_name = None

    def _create_collection(self,
                           collection_name: str,
                           vector_size: int,
                           distance: Distance = Distance.COSINE,
                           payload_schemas: List[PayloadSchema] = None) -> None:
        import pymilvus
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

    def _delete_collection(self, collection_name: str) -> None:
        import pymilvus
        pymilvus.utility.drop_collection(collection_name)

    def _get_collection_info(self, collection_name: str) -> CollectionInfo:
        import pymilvus
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
                              vector_dimension=vector_field.params.get("dim"),
                              distance=distance,
                              payload_schemas=payload_schemas)

    def _has_collection(self, collection_name: str) -> bool:
        import pymilvus
        return pymilvus.utility.has_collection(collection_name)

    def _add(self, point: Point) -> str:
        ids = self._add_all([point])
        return ids[0]

    def _add_all(self, points: List[Point]) -> List[str]:
        # FIXME: add progress bar and batch insert data if the data is too large
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

    def _similarity_search(self,
                           query_vector: Vector,
                           limit: int,
                           score_threshold: Optional[float] = None,
                           criterion: Optional[Criterion] = None,
                           **kwargs: Any) -> List[Point]:
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
            vector = r.entity.get(self._vector_field.name)
            metadata = Metadata()
            for f in payload_field_names:
                v = r.entity.get(f)
                if v is not None:
                    metadata[f] = v
            point = Point(id=r.id,
                          vector=vector,
                          metadata=metadata,
                          score=r.distance)
            # FIXME: filter by score_threshold
            points.append(point)
        return points
