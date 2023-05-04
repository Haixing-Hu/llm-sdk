# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from abc import ABC, abstractmethod
from typing import Any, List, Optional
import logging

from llmsdk.common import Point, Vector


class VectorStore(ABC):
    """
    The interface of vector stores.
    """
    def __init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)

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
               filter: Optional[Any] = None,
               **kwargs: Any) -> List[Point]:
        """
        Searches in the vector store for points whose vector similar to the
        specified vector and satisfies the specified filter.

        :param vector: the specified vector to be searched.
        :param limit: the number of the most similar results to return.
        :param filter: the filter used to filter attributes of points.
        :param kwargs: other vector store specific parameters.
        :return: the list of points as the searching result.
        """

    def close(self) -> None:
        """
        Closes this vector store.
        """
        pass
