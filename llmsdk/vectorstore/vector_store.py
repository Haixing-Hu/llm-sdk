# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from abc import ABC, abstractmethod
from typing import Any, List, Optional
from logging import Logger, getLogger

from ..common import Point, Vector
from ..criterion import Criterion


class VectorStore(ABC):
    """
    The interface of vector stores.
    """
    def __init__(self) -> None:
        self._logger = getLogger(self.__class__.__name__)
        self._collection_name = None
        self._auto_close_connection = False

    @property
    def logger(self) -> Logger:
        return self._logger

    @property
    def collection_name(self) -> str:
        return self._collection_name

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
        self._collection_name = None

    @abstractmethod
    def create_collection(self, collection_name: str) -> None:
        """
        Creates a collection.

        :param collection_name: the name of the collection to be created.
        """

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """
         Deletes a collection.

         :param collection_name: the name of the collection to be deleted.
         """

    @abstractmethod
    def add(self,
            point: Point,
            **kwargs: Any) -> str:
        """
        Adds a point to the vector store.

        :param point: the point to be added. After adding this function, the
            `id` field of this point will be set.
        :param kwargs: other vector store specific parameters.
        :return: the ID of the point added into the vector store.
        """

    def add_all(self,
                points: List[Point],
                **kwargs: Any) -> List[str]:
        """
        Adds all points to the vector store.

        The subclass may override the default implementation of this method for
        optimization.

        :param points: the points to be added. After adding this function, the
            `id` field of each point in this argument will be set.
        :param kwargs: other vector store specific parameters.
        :return: the list of IDs of the vectors added into the vector store.
        """
        result = []
        for point in points:
            id = self.add(point, **kwargs)
            result.append(id)
        return result

    @abstractmethod
    def search(self,
               vector: Vector,
               limit: int,
               criterion: Optional[Criterion] = None,
               **kwargs: Any) -> List[Point]:
        """
        Searches in the vector store for points whose vector similar to the
        specified vector and satisfies the specified filter.

        :param vector: the specified vector to be searched.
        :param limit: the number of the most similar results to return.
        :param criterion: the criterion used to filter attributes of points.
        :param kwargs: other vector store specific parameters.
        :return: the list of points as the searching result.
        """
