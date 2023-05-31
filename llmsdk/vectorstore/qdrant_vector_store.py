# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import Optional, Any, List

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ApiException, UnexpectedResponse

from .payload_schema import PayloadSchema
from .distance import Distance
from .collection_info import CollectionInfo
from .vector_store import VectorStore
from .qdrant_utils import (
    to_qdrant_type,
    to_local_type,
    to_qdrant_distance,
    to_local_distance,
    to_qdrant_point,
    to_local_point,
    criterion_to_filter,
)
from ..common import Vector, Point
from ..criterion import Criterion
from ..generator import IdGenerator


class QdrantVectorStore(VectorStore):
    """
    The vector store based on the Qdrant vector database.
    """

    def __init__(self,
                 client: QdrantClient,
                 id_generator: Optional[IdGenerator] = None) -> None:
        """
        Construct a QdrantVectorStore object.

        To use you should have the ``qdrant-client`` package installed.

        :param client: the qdrant client object.
        :param id_generator: the ID generator used to generate ID of documents.
        """
        super().__init__(id_generator=id_generator)
        self._client = client

    def open(self) -> None:
        self._is_opened = True

    def close(self) -> None:
        self._is_opened = False

    def open_collection(self, collection_name: str) -> None:
        info = self._client.get_collection(collection_name)
        self._collection_name = collection_name
        self._logger.info(f"Successfully opened collection '{collection_name}', "
                          f"which has {info.points_count} points.")

    def close_collection(self) -> None:
        self._collection_name = None

    def has_collection(self, collection_name: str) -> bool:
        try:
            self._client.get_collection(collection_name)
            return True
        except ValueError as e:
            if str(e) == f"Collection {collection_name} not found":
                return False
            raise e
        except UnexpectedResponse as e:
            if e.status_code == 404:
                return False
            raise e
        except ApiException as e:
            raise e

    def create_collection(self,
                          collection_name: str,
                          vector_size: int,
                          distance: Distance = Distance.COSINE,
                          payload_schemas: List[PayloadSchema] = None) -> None:
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

    def add(self, point: Point) -> str:
        self._ensure_collection_opened()
        qdrant_points = [to_qdrant_point(point, self._id_generator)]
        self._logger.debug("Add a point: %s", qdrant_points[0])
        self._client.upsert(collection_name=self._collection_name,
                            points=qdrant_points)
        return point.id

    def add_all(self, points: List[Point]) -> List[str]:
        self._ensure_collection_opened()
        qdrant_points = [to_qdrant_point(pt, self._id_generator) for pt in points]
        self._logger.debug("Add %d points: %s", len(qdrant_points), qdrant_points)
        self._client.upsert(collection_name=self._collection_name,
                            points=qdrant_points)
        self._logger.debug("Totally %d points added.", len(qdrant_points))
        return [p.id for p in points]

    def similarity_search(self,
                          query_vector: Vector,
                          limit: int,
                          criterion: Optional[Criterion] = None,
                          **kwargs: Any) -> List[Point]:
        self._ensure_collection_opened()
        query_filter = criterion_to_filter(criterion)
        self._logger.debug("Search: vector=%s, limit=%d, filter=%s",
                           query_vector, limit, query_filter)
        points = self._client.search(collection_name=self._collection_name,
                                     query_vector=query_vector,
                                     query_filter=query_filter,
                                     limit=limit,
                                     with_vectors=True,
                                     **kwargs)
        self._logger.debug("Search result: %s", points)
        return [to_local_point(p) for p in points]
