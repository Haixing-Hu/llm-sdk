# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from abc import ABC, abstractmethod
from typing import Any, List, Optional
from logging import Logger, getLogger

from ..common import Point, Vector, SearchType
from ..criterion import Criterion
from ..generator import IdGenerator, Uuid4Generator
from .distance import Distance
from .payload_schema import PayloadSchema
from .collection_info import CollectionInfo
from .vector_store_utils import maximal_marginal_relevance


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

    def search(self,
               query_vector: Vector,
               limit: int,
               criterion: Optional[Criterion] = None,
               search_type: SearchType = SearchType.SIMILARITY,
               **kwargs: Any) -> List[Point]:
        """
        Searches in the vector store for the specified points.

        :param query_vector: the specified vector to be searched.
        :param limit: the number of the most similar results to return.
        :param criterion: the criterion used to filter attributes of points.
        :param search_type: the type of searching.
        :param kwargs: other arguments.
        :return: the list of points as the searching result.
        """
        self._ensure_collection_opened()
        match search_type:
            case SearchType.SIMILARITY:
                return self.similarity_search(
                    query_vector=query_vector,
                    limit=limit,
                    criterion=criterion,
                    **kwargs
                )
            case SearchType.MAX_MARGINAL_RELEVANCE:
                return self.max_marginal_relevance_search(
                    query_vector=query_vector,
                    limit=limit,
                    criterion=criterion,
                    **kwargs
                )
            case _:
                raise ValueError(f"Unsupported search type: {search_type}")

    @abstractmethod
    def similarity_search(self,
                          query_vector: Vector,
                          limit: int,
                          criterion: Optional[Criterion] = None,
                          **kwargs: Any) -> List[Point]:
        """
        Searches in the vector store for points whose vector is similar to the
        specified vector and satisfies the specified filter.

        :param query_vector: the specified vector to be searched.
        :param limit: the number of the most similar results to return.
        :param criterion: the criterion used to filter attributes of points.
        :param kwargs: other arguments.
        :return: the list of points as the searching result.
        """

    def max_marginal_relevance_search(self,
                                      query_vector: Vector,
                                      limit: int,
                                      criterion: Optional[Criterion] = None,
                                      fetch_limit: int = 20,
                                      lambda_multiply: float = 0.5,
                                      **kwargs: Any) -> List[Point]:
        """
        Searches in the vector store for points whose vector similar to the
        specified vector and satisfies the specified filter, using the maximal
        marginal relevance.

        :param query_vector: the specified vector to be searched.
        :param limit: the number of the most similar results to return.
        :param criterion: the criterion used to filter attributes of points.
        :param fetch_limit: the number of documents to fetch to pass to MMR
            algorithm.
        :param lambda_multiply: a number between 0 and 1 that determines the
            degree of diversity among the result, with 0 corresponding to the
            maximum diversity and 1 to minimum diversity. Defaults to 0.5.
        :param kwargs: other arguments.
        :return: the list of points as the searching result.
        """
        self._ensure_collection_opened()
        result = self.similarity_search(query_vector=query_vector,
                                        limit=fetch_limit,
                                        criterion=criterion,
                                        **kwargs)
        similarity_vectors = [p.vector for p in result]
        mmr_selected = maximal_marginal_relevance(
            query_vector=query_vector,
            similarity_vectors=similarity_vectors,
            limit=limit,
            lambda_multipy=lambda_multiply
        )
        return [result[i] for i in mmr_selected]

    def _ensure_collection_opened(self):
        """
        Ensure that this store has opened a collection.
        :raise RuntimeError: if no collection was opened for this vector store.
        """
        if self._collection_name is None:
            raise RuntimeError("No collection was opened for this vector store.")
