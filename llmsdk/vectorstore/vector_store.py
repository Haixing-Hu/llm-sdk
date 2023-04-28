# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional

from llmsdk.common import Point, Vector


class VectorStore(ABC):
    """
    The interface of vector stores.
    """

    @abstractmethod
    def add(self,
            vector: Vector,
            **kwargs: Any) -> str:
        """
        Adds a vector to the vector store.

        :param vector: the vector to be added. After adding, the `id` field of
            this argument will be set.
        :param kwargs: other vector store specific parameters.
        :return: the ID of the vector added into the vector store.
        """

    def add_all(self,
                vectors: Iterable[Vector],
                **kwargs: Any) -> List[str]:
        """
        Adds a vector to the vector store.

        The subclass may override the default implementation of this method for
        optimization.

        :param vectors: the vectors to be added, the `id` field of vectors in
            this argument will be set.
        :param kwargs: other vector store specific parameters.
        :return: the list of IDs of the vectors added into the vector store.
        """
        result = []
        for vector in vectors:
            id = self.add(vector, **kwargs)
            result.append(id)
        return result

    @abstractmethod
    def search(self,
               query: Point,
               limit: int,
               filter: Optional[Any] = None,
               **kwargs: Any) -> List[Vector]:
        """
        Searches in the vector store for the vectors similar to the specified
        point.

        :param query: the specified point to be searched.
        :param limit: the number of the most similar results to return.
        :param filter: the filter used to filter attributes of searching results.
        :param kwargs: other vector store specific parameters.
        :return: the list of search results.
        """
