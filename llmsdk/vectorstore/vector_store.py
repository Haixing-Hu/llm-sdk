# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from logging import Logger, getLogger

from ..common.distance import Distance
from ..common.vector import Vector
from ..common.point import Point
from ..common.search_type import SearchType
from ..criterion.criterion import Criterion
from ..generator.id_generator import IdGenerator
from ..generator.default_id_generator import DefaultIdGenerator
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
        self._store_name = self.__class__.__name__
        self._id_generator = id_generator or DefaultIdGenerator()
        self._is_opened = False
        self._collection_name = None

    @property
    def logger(self) -> Logger:
        return self._logger

    @property
    def store_name(self) -> str:
        return self._store_name

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

    @property
    def is_collection_opened(self) -> bool:
        """
        Tests whether a collection of this vector store is opened.

        :return: True if a collection of this vector store is opened; False
            otherwise.
        """
        return self._collection_name is not None

    def open(self, **kwargs: Any) -> None:
        """
        Opens this vector store.

        :param kwargs: the arguments used to open this vector store.
        """
        self._logger.info("Opening the %s...", self._store_name)
        self._ensure_store_closed()
        self._open(**kwargs)
        self._logger.info("Successfully opened the %s.", self._store_name)

    @abstractmethod
    def _open(self, **kwargs: Any) -> None:
        """
        Opens this vector store.

        This method should be implemented by the subclasses. The implementation
        do not have to check the state of this vector store.

        :param kwargs: the arguments used to open this vector store.
        """

    def close(self) -> None:
        """
        Closes this vector store.

        If any collection of this vector store is opened, it will be closed
        firstly.

        Closes a closed vector store have no effect.
        """
        if self.is_opened:
            self._logger.info("Closing the %s...", self._store_name)
            self._close()
            self._logger.info("Successfully closed the %s.", self._store_name)

    @abstractmethod
    def _close(self) -> None:
        """
        Closes this vector store.

        If any collection of this vector store is opened, it will be closed
        firstly.

        This method should be implemented by the subclasses. The implementation
        do not have to check the state of this vector store.
        """

    def open_collection(self, collection_name: str) -> None:
        """
        Opens the specified collection, and sets it as the current collection.

        :param collection_name: the name of the specified collection.
        """
        self._logger.info("Opening the collection '%s'...", collection_name)
        self._ensure_store_opened()
        self._ensure_collection_closed()
        self._open_collection(collection_name)
        self._logger.info("Successfully opened collection '%s'.", collection_name)

    @abstractmethod
    def _open_collection(self, collection_name: str) -> None:
        """
        Opens the specified collection, and sets it as the current collection.

        This method should be implemented by the subclasses. The implementation
        do not have to check the state of this vector store.

        :param collection_name: the name of the specified collection.
        """

    def close_collection(self) -> None:
        """
        close the current collection.

        Closes a closed collection have no effect.
        """
        if self.is_collection_opened:
            collection_name = self._collection_name
            self._logger.info("Closing the collection '%s'...", collection_name)
            self._ensure_store_opened()
            self._close_collection()
            self._logger.info("Successfully closed collection '%s'.", collection_name)

    @abstractmethod
    def _close_collection(self) -> None:
        """
        close the current collection.

        This method should be implemented by the subclasses. The implementation
        do not have to check the state of this vector store.
        """

    def has_collection(self, collection_name: str) -> bool:
        """
        Tests whether this vector store has a collection with the specified name.

        :param collection_name: the specified collection name.
        :return: True if this vector store has a collection with the specified
            name; False otherwise.
        """
        self._logger.info("Testing the existence of the collection '%s'...",
                          collection_name)
        self._ensure_store_opened()
        result = self._has_collection(collection_name)
        self._logger.info("The collection '%s' %s exist.", collection_name,
                          "does" if result else "does not")
        return result

    @abstractmethod
    def _has_collection(self, collection_name: str) -> bool:
        """
        Tests whether this vector store has a collection with the specified name.

        This method should be implemented by the subclasses. The implementation
        do not have to check the state of this vector store.

        :param collection_name: the specified collection name.
        :return: True if this vector store has a collection with the specified
            name; False otherwise.
        """

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
        self._logger.info("Creating the new collection '%s'...", collection_name)
        self._ensure_store_opened()
        self._create_collection(collection_name, vector_size, distance, payload_schemas)
        self._logger.info("Successfully created the collection '%s'.", collection_name)

    @abstractmethod
    def _create_collection(self,
                           collection_name: str,
                           vector_size: int,
                           distance: Distance = Distance.COSINE,
                           payload_schemas: List[PayloadSchema] = None) -> None:
        """
        Creates a collection.

        This method should be implemented by the subclasses. The implementation
        do not have to check the state of this vector store.

        :param collection_name: the name of the collection to be created.
        :param vector_size: the size of vectors stored in the new collection.
        :param distance: the distance used to estimate the similarity of vectors
            with each other.
        :param payload_schemas: the list of payload field schemas of the new
            collection.
        """

    def delete_collection(self, collection_name: str) -> None:
        """
        Deletes a collection.

        Deleting the currently opened collection will case an exception.

        :param collection_name: the name of the collection to be deleted.
        """
        self._logger.info("Delete the collection '%s' ...", collection_name)
        self._ensure_store_opened()
        if self._collection_name == collection_name:
            raise ValueError(f"Can not delete opened collection "
                             f"'{collection_name}'. You must close it before "
                             f"deleting it.")
        self._delete_collection(collection_name)
        self._logger.info("Successfully deleted the collection '%s'.",
                          collection_name)

    @abstractmethod
    def _delete_collection(self, collection_name: str) -> None:
        """
        Deletes a collection.

        This method should be implemented by the subclasses. The implementation
        do not have to check the state of this vector store.

        :param collection_name: the name of the collection to be deleted.
        """

    def get_collection_info(self, collection_name: str) -> CollectionInfo:
        """
        Gets the information of the specified collection.

        :param collection_name: the name of the specified collection.
        :return: the information of the specified collection.
        """
        self._logger.info("Getting the information of the collection '%s'...",
                          collection_name)
        self._ensure_store_opened()
        info = self._get_collection_info(collection_name)
        self._logger.info("Successfully got the information of the collection '%s': %s",
                          collection_name, info)
        return info

    @abstractmethod
    def _get_collection_info(self, collection_name: str) -> CollectionInfo:
        """
        Gets the information of the specified collection.

        This method should be implemented by the subclasses. The implementation
        do not have to check the state of this vector store.

        :param collection_name: the name of the specified collection.
        :return: the information of the specified collection.
        """

    def add(self, point: Point) -> str:
        """
        Adds a point to the vector store.

        :param point: the point to be added. After adding this function, the
            `id` field of this point will be set.
        :return: the ID of the point added into the vector store.
        """
        self._logger.info("Adding a point to the collection '%s'...",
                          self._collection_name)
        self._logger.debug("The point to add is: %s", point)
        self._ensure_store_opened()
        self._ensure_collection_opened()
        result = self._add(point)
        self._logger.info("Successfully added the point to the collection '%s'.",
                          self._collection_name)
        self._logger.debug("The ID of the point added is: %s", result)
        return result

    @abstractmethod
    def _add(self, point: Point) -> str:
        """
        Adds a point to the vector store.

        This method should be implemented by the subclasses. The implementation
        do not have to check the state of this vector store.

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
        self._logger.info("Adding %d points to the collection '%s'...",
                          len(points), self._collection_name)
        self._logger.debug("The points to add are: %s", points)
        self._ensure_store_opened()
        self._ensure_collection_opened()
        result = self._add_all(points)
        self._logger.info("Successfully added %d point to the collection '%s'.",
                          len(points), self._collection_name)
        self._logger.debug("The IDs of the points added are: %s", result)
        return result

    def _add_all(self, points: List[Point]) -> List[str]:
        """
        Adds all points to the vector store.

        The subclass may override the default implementation of this method for
        optimization.

        :param points: the points to be added. After adding this function, the
            `id` field of each point in this argument will be set.
        :return: the list of IDs of the vectors added into the vector store.
        """
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
        self._logger.info("Performing similarity search ...")
        self._logger.debug("query_vector=%s, limit=%d, score_threshold=%f, "
                           "criterion = %s", query_vector, limit,
                           score_threshold, criterion)
        self._ensure_store_opened()
        self._ensure_collection_opened()
        result = self._similarity_search(query_vector=query_vector,
                                         limit=limit,
                                         score_threshold=score_threshold,
                                         criterion=criterion,
                                         **kwargs)
        self._logger.info("Successfully performed similarity search.")
        self._logger.debug("Searching result are: %s", result)
        return result

    @abstractmethod
    def _similarity_search(self,
                           query_vector: Vector,
                           limit: int,
                           score_threshold: Optional[float] = None,
                           criterion: Optional[Criterion] = None,
                           **kwargs: Any) -> List[Point]:
        """
        Searches in the vector store for points whose vector is similar to the
        specified vector and satisfies the specified filter.

        This method should be implemented by the subclasses. The implementation
        do not have to check the state of this vector store.

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
        self._logger.info("Performing max marginal relevance search ...")
        self._logger.debug("query_vector=%s, limit=%d, score_threshold=%f, "
                           "criterion = %s, fetch_limit=%d, lambda_multiply=%f",
                           query_vector, limit, score_threshold, criterion,
                           fetch_limit, lambda_multiply)
        self._ensure_store_opened()
        self._ensure_collection_opened()
        result = self._max_marginal_relevance_search(
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            criterion=criterion,
            fetch_limit=fetch_limit,
            lambda_multiply=lambda_multiply,
            **kwargs
        )
        self._logger.info("Successfully found %d points.", len(result))
        self._logger.debug("Searching result are: %s", result)
        return result

    def _max_marginal_relevance_search(self,
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

        The subclass may override the default implementation of this method for
        optimization.

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

    def _ensure_store_opened(self):
        """
        Ensure this store is opened.

        :raise RuntimeError: if this vector store is not opened.
        """
        if not self._is_opened:
            raise RuntimeError("This vector store is not opened.")

    def _ensure_store_closed(self):
        """
        Ensure this store is closed.

        :raise RuntimeError: if this vector store is not closed.
        """
        if self._is_opened:
            raise RuntimeError("This vector store is not closed.")

    def _ensure_collection_opened(self):
        """
        Ensure a collection of this store is opened.

        :raise RuntimeError: if no collection in this vector store was opened.
        """
        if self._collection_name is None:
            raise RuntimeError("No collection of this vector store was opened.")

    def _ensure_collection_closed(self):
        """
        Ensure the all collections in this store is closed.

        :raise RuntimeError: if any collection in this vector store was opened.
        """
        if self._collection_name is not None:
            raise RuntimeError("A collection of this vector store was still opened.")
