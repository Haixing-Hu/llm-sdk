# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from abc import ABC, abstractmethod
from typing import Any, List, Optional
from logging import Logger, getLogger

from .distance import Distance
from .payload_schema import PayloadSchema
from .collection_info import CollectionInfo
from ..common import Point, Vector
from ..criterion import Criterion
from ..generator import IdGenerator, Uuid4Generator


class VectorStore(ABC):
    """
    The interface of vector stores.
    """

    def __init__(self, id_generator: Optional[IdGenerator] = None) -> None:
        """
        Constructs a vector store.

        :param id_generator: the ID generator used to generate ID of documents.
        """
        self._logger = getLogger(self.__class__.__name__)
        self._collection_name = None
        self._auto_close_connection = False
        self._id_generator = id_generator or Uuid4Generator()

    @property
    def logger(self) -> Logger:
        return self._logger

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def id_generator(self) -> IdGenerator:
        return self._id_generator

    def open(self) -> None:
        """
        Opens this vector store.
        """
        pass

    def close(self) -> None:
        """
        Closes this vector store.
        """
        pass

    def open_collection(self, collection_name: str) -> None:
        """
        Opens the specified collection, and sets it as the current collection.

        :param collection_name: the name of the specified collection.
        """
        self.close_collection()
        self._collection_name = collection_name

    def close_collection(self) -> None:
        """
        close the current collection.
        """
        self._ensure_collection_opened()
        self._collection_name = None

    @abstractmethod
    def create_collection(self,
                          collection_name: str,
                          vector_size: int,
                          distance: Distance = Distance.COSINE,
                          payload_schemas: List[PayloadSchema] = None) -> None:
        """
        Creates a collection.

        :param collection_name: the name of the collection to be created.
        :param vector_size: the size of vectors stored in the new collection.
        :param distance: the distance used to estimate the similarity of vectors
            with each other.
        :param payload_schemas: the list of payload field schemas of the new
            collection.
        """

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """
         Deletes a collection.

         :param collection_name: the name of the collection to be deleted.
         """

    @abstractmethod
    def get_collection_info(self, collection_name: str) -> CollectionInfo:
        """
        Gets the information of the specified collection.

        :param collection_name: the name of the specified collection.
        :return: the information of the specified collection.
        """

    @abstractmethod
    def add(self, point: Point) -> str:
        """
        Adds a point to the vector store.

        :param point: the point to be added. After adding this function, the
            `id` field of this point will be set.
        :return: the ID of the point added into the vector store.
        """

    def add_all(self, points: List[Point]) -> List[str]:
        """
        Adds all points to the vector store.

        The subclass may override the default implementation of this method for
        optimization.

        :param points: the points to be added. After adding this function, the
            `id` field of each point in this argument will be set.
        :return: the list of IDs of the vectors added into the vector store.
        """
        self._ensure_collection_opened()
        result = []
        for point in points:
            id = self.add(point)
            result.append(id)
        return result

    @abstractmethod
    def search(self,
               vector: Vector,
               limit: int,
               criterion: Optional[Criterion] = None) -> List[Point]:
        """
        Searches in the vector store for points whose vector similar to the
        specified vector and satisfies the specified filter.

        :param vector: the specified vector to be searched.
        :param limit: the number of the most similar results to return.
        :param criterion: the criterion used to filter attributes of points.
        :return: the list of points as the searching result.
        """

    def _ensure_collection_opened(self):
        """
        Ensure that this store has opened a collection.
        :raise RuntimeError: if no collection was opened for this vector store.
        """
        if self._collection_name is None:
            raise RuntimeError("No collection was opened for this vector store.")
