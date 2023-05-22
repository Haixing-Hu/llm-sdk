# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import Optional, Any, List

from qdrant_client import QdrantClient
from qdrant_client.http import models

from .payload_schema import PayloadSchema
from .distance import Distance
from .collection_info import CollectionInfo
from .vector_store import VectorStore
from .qdrant_utils import to_qdrant_type, to_local_type
from .qdrant_utils import to_qdrant_distance, to_local_distance
from .qdrant_utils import to_qdrant_point, to_local_point
from .qdrant_utils import criterion_to_filter
from ..common import Vector, Point
from ..criterion import Criterion


class QdrantVectorStore(VectorStore):
    """
    The vector store based on the Qdrant vector database.
    """

    def __init__(self,
                 client: QdrantClient) -> None:
        """
        Construct a QdrantVectorStore object.

        To use you should have the ``qdrant-client`` package installed.

        :param client: the qdrant client object.
        """
        super().__init__()
        self._client = client

    def create_collection(self,
                          collection_name: str,
                          vector_size: int,
                          payload_schemas: List[PayloadSchema] = None,
                          distance: Distance = Distance.COSINE) -> None:
        config = models.VectorParams(size=vector_size,
                                     distance=to_qdrant_distance(distance))
        self._logger.debug("Create a collection: name=%s, config={%s}",
                           collection_name, config)
        self._client.create_collection(collection_name=collection_name,
                                       vectors_config=config)
        if payload_schemas is not None:
            for schema in payload_schemas:
                payload_schema = to_qdrant_type(schema.type)
                self._logger.debug("Create a payload index: collection=%s, "
                                   "field_name=%s, field_schema=%s",
                                   collection_name, schema.name, payload_schema)
                self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name=schema.name,
                    field_schema=payload_schema
                )
        pass

    def delete_collection(self, collection_name: str) -> None:
        self._logger.debug("Delete the collection: %s", collection_name)
        self._client.delete_collection(collection_name)

    def get_collection_info(self, collection_name: str) -> CollectionInfo:
        info = self._client.get_collection(collection_name)
        vector_size = info.config.params.vectors.size
        distance = to_local_distance(info.config.params.vectors.distance)
        payload_schemas = [
            PayloadSchema(name=k, type=to_local_type(v.data_type))
            for k, v in info.payload_schema.items()
        ]
        return CollectionInfo(name=collection_name,
                              size=info.points_count,
                              vector_size=vector_size,
                              distance=distance,
                              payload_schemas=payload_schemas)

    def add(self, point: Point, **kwargs: Any) -> str:
        qdrant_points = [to_qdrant_point(point)]
        self._logger.debug("Add a point: %s", qdrant_points[0])
        self._client.upsert(collection_name=self._collection_name,
                            points=qdrant_points,
                            **kwargs)
        return point.id

    def add_all(self, points: List[Point], **kwargs: Any) -> List[str]:
        qdrant_points = [to_qdrant_point(pt) for pt in points]
        self._logger.debug("Add points: %s", qdrant_points)
        self._client.upsert(collection_name=self._collection_name,
                            points=qdrant_points,
                            **kwargs)
        return [p.id for p in points]

    def search(self,
               vector: Vector,
               limit: int,
               criterion: Optional[Criterion] = None,
               **kwargs: Any) -> List[Point]:
        query_filter = criterion_to_filter(criterion)
        self._logger.debug("Search: vector=%s, limit=%d, filter=%s",
                           vector, limit, query_filter)
        points = self._client.search(collection_name=self._collection_name,
                                     query_filter=query_filter,
                                     query_vector=vector,
                                     limit=limit,
                                     with_vectors=True,
                                     **kwargs)
        self._logger.debug("Search result: %s", points)
        return [to_local_point(p) for p in points]
