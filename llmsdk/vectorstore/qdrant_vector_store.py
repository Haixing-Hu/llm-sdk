# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import Optional, Any, List, Iterable
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, ScoredPoint

from .vector_store import VectorStore
from ..common import Vector, Point


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


class QdrantVectorStore(VectorStore):
    """
    The vector store based on the Qdrant vector database.
    """
    def __init__(self,
                 client: QdrantClient,
                 collection: str) -> None:
        """
        Construct a QdrantVectorStore object.

        To use you should have the ``qdrant-client`` package installed.

        :param client: the qdrant client object.
        :param collection: the name of the collection.
        """
        self._client = client
        self._collection = collection

    def add(self,
            point: Point,
            **kwargs: Any) -> str:
        struct = point_to_point_struct(point)
        self._client.upsert(collection_name=self._collection,
                            points=[struct],
                            **kwargs)
        return point.id

    def add_all(self,
                points: Iterable[Point],
                **kwargs: Any) -> List[str]:
        structs = [point_to_point_struct(point) for point in points]
        self._client.upsert(collection_name=self._collection,
                            points=structs,
                            **kwargs)
        return [point.id for point in points]

    def search(self,
               vector: Vector,
               limit: int,
               filter: Optional[Any] = None,
               **kwargs: Any) -> List[Point]:
        result = self._client.search(collection_name=self._collection,
                                     query_filter=filter,
                                     query_vector=vector,
                                     limit=limit,
                                     **kwargs)
        return [scored_point_to_point(point) for point in result]
