# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from abc import ABC, abstractmethod
from typing import Any, List, Optional
from logging import Logger, getLogger

from ..common import Point, Vector, SearchType, Distance
from ..criterion import Criterion
from ..generator import IdGenerator, DefaultIdGenerator
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
        self._id_generator = id_generator or DefaultIdGenerator()
        self._is_opened = False
        self._collection_name = None
        self._auto_close_connection = False

    @property
    def logger(self) -> Logger:
        return self._logger

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def id_generator(self) -> IdGenerator:
        return self._id_generator

    @property
    def is_opened(self) -> bool:
        """
        Tests whether this vector store is opened.

        :return: True if this vector store is opened; False otherwise.
        """
        return self._is_opened

    @abstractmethod
    def open(self) -> None:
        """
        Opens this vector store.
        """

    @abstractmethod
    def close(self) -> None:
        """
        Closes this vector store.
        """

    @property
    def is_collection_opened(self) -> bool:
        """
        Tests whether a collection of this vector store is opened.

        :return: True if a collection of this vector store is opened; False
            otherwise.
        """
        return self._collection_name is not None

    @abstractmethod
    def open_collection(self, collection_name: str) -> None:
        """
        Opens the specified collection, and sets it as the current collection.

        :param collection_name: the name of the specified collection.
        """

    @abstractmethod
    def close_collection(self) -> None:
        """
        close the current collection.
        """

    @abstractmethod
    def has_collection(self, collection_name: str) -> bool:
        """
        Tests whether this vector store has a collection with the specified name.

        :param collection_name: the specified collection name.
        :return: True if this vector store has a collection with the specified
            name; False otherwise.
        """

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
               score_threshold: Optional[float] = None,
               criterion: Optional[Criterion] = None,
               search_type: SearchType = SearchType.SIMILARITY,
               **kwargs: Any) -> List[Point]:
        """
        Searches in the vector store for the specified points.

        :param query_vector: the specified vector to be searched.
        :param limit: the number of the most similar results to return.
        :param score_threshold: indicates the minimal score threshold for the
            result. If provided, less similar results will not be returned.
            Score of the returned result might be higher or smaller than the
            threshold depending on the Distance function used. E.g. for cosine
            similarity only higher scores will be returned.
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
                    score_threshold=score_threshold,
                    criterion=criterion,
                    **kwargs
                )
            case SearchType.MAX_MARGINAL_RELEVANCE:
                return self.max_marginal_relevance_search(
                    query_vector=query_vector,
                    limit=limit,
                    score_threshold=score_threshold,
                    criterion=criterion,
                    **kwargs
                )
            case _:
                raise ValueError(f"Unsupported search type: {search_type}")

    @abstractmethod
    def similarity_search(self,
                          query_vector: Vector,
                          limit: int,
                          score_threshold: Optional[float] = None,
                          criterion: Optional[Criterion] = None,
                          **kwargs: Any) -> List[Point]:
        """
        Searches in the vector store for points whose vector is similar to the
        specified vector and satisfies the specified filter.

        :param query_vector: the specified vector to be searched.
        :param limit: the number of the most similar results to return.
        :param score_threshold: indicates the minimal score threshold for the
            result. If provided, less similar results will not be returned.
            Score of the returned result might be higher or smaller than the
            threshold depending on the Distance function used. E.g. for cosine
            similarity only higher scores will be returned.
        :param criterion: the criterion used to filter attributes of points.
        :param kwargs: other arguments.
        :return: the list of points as the searching result.
        """

    def max_marginal_relevance_search(self,
                                      query_vector: Vector,
                                      limit: int,
                                      score_threshold: Optional[float] = None,
                                      criterion: Optional[Criterion] = None,
                                      fetch_limit: int = None,
                                      lambda_multiply: float = 0.5,
                                      **kwargs: Any) -> List[Point]:
        """
        Searches in the vector store for points whose vector similar to the
        specified vector and satisfies the specified filter, using the maximal
        marginal relevance.

        :param query_vector: the specified vector to be searched.
        :param limit: the number of the most similar results to return.
        :param score_threshold: indicates the minimal score threshold for the
            result. If provided, less similar results will not be returned.
            Score of the returned result might be higher or smaller than the
            threshold depending on the Distance function used. E.g. for cosine
            similarity only higher scores will be returned.
        :param criterion: the criterion used to filter attributes of points.
        :param fetch_limit: the number of documents to fetch to pass to MMR
            algorithm. If this argument is set to None, the function will use
            5 times of limit for the fetch limit.
        :param lambda_multiply: a number between 0 and 1 that determines the
            degree of diversity among the result, with 0 corresponding to the
            maximum diversity and 1 to minimum diversity. Defaults to 0.5.
        :param kwargs: other arguments.
        :return: the list of points as the searching result.
        """
        self._ensure_collection_opened()
        if fetch_limit is None:
            fetch_limit = 5 * limit
        result = self.similarity_search(query_vector=query_vector,
                                        limit=fetch_limit,
                                        score_threshold=score_threshold,
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
        Ensure this store is opened and a collection of this store is opened.

        :raise RuntimeError: if this vector store is not opened, or no
            collection was opened for this vector store.
        """
        if not self._is_opened:
            raise RuntimeError("This vector store is not opened.")
        if self._collection_name is None:
            raise RuntimeError("No collection was opened for this vector store.")
