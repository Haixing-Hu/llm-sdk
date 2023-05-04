# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import Optional, Any, List
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, ScoredPoint

from .vector_store import VectorStore
from ..common import Vector, Point


class QdrantVectorStore(VectorStore):
    """
    The vector store based on the Qdrant vector database.
    """

    def __init__(self,
                 client: QdrantClient,
                 collection_name: str) -> None:
        """
        Construct a QdrantVectorStore object.

        To use you should have the ``qdrant-client`` package installed.

        :param client: the qdrant client object.
        :param collection_name: the name of the collection.
        """
        super().__init__()
        self._client = client
        self._collection_name = collection_name

    def add(self,
            point: Point,
            **kwargs: Any) -> str:
        structs = [point_to_point_struct(point)]
        self._logger.debug("Insert points: %s", structs)
        self._client.upsert(collection_name=self._collection_name,
                            points=structs,
                            **kwargs)
        return point.id

    def add_all(self,
                points: List[Point],
                **kwargs: Any) -> List[str]:
        structs = [point_to_point_struct(pt) for pt in points]
        self._logger.debug("Insert points: %s", structs)
        self._client.upsert(collection_name=self._collection_name,
                            points=structs,
                            **kwargs)
        return [p.id for p in points]

    def search(self,
               vector: Vector,
               limit: int,
               filter: Optional[Any] = None,
               **kwargs: Any) -> List[Point]:
        points = self._client.search(collection_name=self._collection_name,
                                     query_filter=filter,
                                     query_vector=vector,
                                     limit=limit,
                                     **kwargs)
        return [scored_point_to_point(p) for p in points]


def point_to_point_struct(point: Point) -> PointStruct:
    """
    Converts a Point object into a qdrant PointStruct object.

    :param point: a Point object.
    :return: the converted PointStruct object.
    """
    if point.id is None:
        point.id = uuid.uuid4()
    return PointStruct(id=point.id,
                       vector=point.vector,
                       payload=point.metadata)


def scored_point_to_point(scored_point: ScoredPoint) -> Point:
    """
    Converts a qdrant ScoredPoint object into a Point object.

    :param scored_point: a qdrant ScoredPoint object.
    :return: the converted Point object.
    """
    return Point(id=scored_point.id,
                 vector=scored_point.vector,
                 metadata=scored_point.payload,
                 score=scored_point.score)
