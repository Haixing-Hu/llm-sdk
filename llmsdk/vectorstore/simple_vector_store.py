# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import copy
from typing import Optional, Any, List, Dict

from . import CollectionInfo, PayloadSchema
from .vector_store import VectorStore
from ..common import Vector, Point, Distance
from ..criterion import Criterion


class SimpleVectorStore(VectorStore):
    """
    A simple implementation of vector store.
    """

    def __init__(self):
        super().__init__()
        self._collections: Dict[str, List[Point]] = {}
        self._collections_info: Dict[str, CollectionInfo] = {}

    def open(self) -> None:
        self._is_opened = True

    def close(self) -> None:
        self._collection_name = None
        self._is_opened = False
        self._collections = {}
        self._collections_info = {}

    def open_collection(self, collection_name: str) -> None:
        if collection_name in self._collections:
            self._collection_name = collection_name
        else:
            raise ValueError(f"No such collection '{collection_name}'.")

    def close_collection(self) -> None:
        self._collection_name = None

    def has_collection(self, collection_name: str) -> bool:
        return collection_name in self._collections

    def create_collection(self, collection_name: str,
                          vector_size: int,
                          distance: Distance = Distance.COSINE,
                          payload_schemas: List[PayloadSchema] = None) -> None:
        if collection_name in self._collections_info:
            raise ValueError(f"The collection '{collection_name}' already exist.")
        info = CollectionInfo(name=collection_name,
                              size=0,
                              vector_size=vector_size,
                              distance=distance,
                              payload_schemas=payload_schemas)
        self._collections_info[collection_name] = info
        self._collections[collection_name] = []

    def delete_collection(self, collection_name: str) -> None:
        if collection_name in self._collections_info:
            self._collections.pop(collection_name)
            self._collections_info.pop(collection_name)
            if self._collection_name == collection_name:
                self._collection_name = None
        else:
            raise ValueError(f"No such collection '{collection_name}'.")

    def get_collection_info(self, collection_name: str) -> CollectionInfo:
        if collection_name in self._collections_info:
            return self._collections_info[collection_name]
        else:
            raise ValueError(f"No such collection '{collection_name}'.")

    def add(self, point: Point) -> str:
        self._ensure_collection_opened()
        collection = self._collections[self._collection_name]
        info = self._collections_info[self._collection_name]
        if not point.id:
            point.id = self._id_generator.generate()
        collection.append(copy.deepcopy(point))
        new_info = CollectionInfo(name=info.name,
                                  size=info.size + 1,
                                  vector_size=info.vector_size,
                                  distance=info.distance,
                                  payload_schemas=info.payload_schemas)
        self._collections_info[self._collection_name] = new_info
        return point.id

    def similarity_search(self,
                          query_vector: Vector,
                          limit: int,
                          criterion: Optional[Criterion] = None,
                          **kwargs: Any) -> List[Point]:
        self._ensure_collection_opened()
        collection = self._collections[self._collection_name]
        info = self._collections_info[self._collection_name]
        points = self._filter_points(collection, criterion)
        for p in points:
            p.score = info.distance.between(p.vector, query_vector)
        result = sorted(points, key=lambda p: p.score, reverse=True)
        return result[:limit]

    def _filter_points(self,
                       collection: List[Point],
                       criterion: Optional[Criterion]) -> List[Point]:
        # TODO
        return [copy.deepcopy(p) for p in collection]
