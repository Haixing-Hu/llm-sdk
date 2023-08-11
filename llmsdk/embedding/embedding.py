# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from abc import ABC, abstractmethod
from typing import Any, List, Optional
from logging import Logger, getLogger

from cachetools import Cache, LRUCache
from tqdm import tqdm

from ..common.document import Document
from ..common.vector import Vector
from ..common.point import Point
from ..generator.id_generator import IdGenerator
from ..generator.default_id_generator import DefaultIdGenerator


class Embedding(ABC):
    """
    The interface of sentence embedding models.
    """

    DEFAULT_USE_CACHE: bool = True
    """
    The default value of the flag indicating whether to use the cache.
    """

    DEFAULT_CACHE_SIZE: int = 10000
    """
    The default number of text embeddings to be cached.
    """

    DEFAULT_SHOW_PROGRESS: bool = False
    """
    The default value of the flag indicating whether to show the progress of 
    embedding.
    """

    DEFAULT_MIN_SIZE_TO_SHOW_PROGRESS: int = 100
    """
    The default minimum number of embedding texts to show the embedding progress.
    """

    def __init__(self,
                 vector_dimension: int,
                 id_generator: Optional[IdGenerator] = None,
                 use_cache: bool = DEFAULT_USE_CACHE,
                 cache_size: int = DEFAULT_CACHE_SIZE,
                 show_progress: bool = DEFAULT_SHOW_PROGRESS,
                 min_size_to_show_progress: int = DEFAULT_MIN_SIZE_TO_SHOW_PROGRESS) -> None:
        """
        Creates an Embedding object.

        :param vector_dimension: the number of dimension of the embedded vectors.
        :param id_generator: the generator used to generating document IDs.
        :param use_cache: indicates whether to use the cache to store the
            embedded vectors of texts. If this argument is True, the embedded
            vectors of texts will be cached in a LRU cache. Otherwise, the
            embedded vectors of texts will not be cached.
        :param cache_size: the number of text embeddings to be cached. This
            argument is ignored if the use_cache argument is False.
        :param show_progress: indicates whether to show the progress of
            embedding.
        :param min_size_to_show_progress: the minimum number of embedding texts
            to show the embedding progress.
        """
        self._logger = getLogger(self.__class__.__name__)
        self._vector_dimension = vector_dimension
        self._id_generator = id_generator or DefaultIdGenerator()
        if use_cache:
            self._cache = LRUCache(maxsize=cache_size)
        else:
            self._cache = None
        self._show_progress = show_progress
        self._min_size_to_show_progress = min_size_to_show_progress

    @property
    def logger(self) -> Logger:
        """
        Gets the logger of this object.

        :return: the logger of this object.
        """
        return self._logger

    def set_logging_level(self, level: int | str) -> None:
        """
        Sets the logging level of this object.

        :param level: the logging level to be set.
        """
        self._logger.setLevel(level)

    @property
    def vector_dimension(self) -> int:
        """
        Gets the number of dimensions of the embedded vectors.

        :return: the number of dimensions of the embedded vectors.
        """
        return self._vector_dimension

    @property
    def id_generator(self) -> IdGenerator:
        return self._id_generator

    @property
    def cache(self) -> Optional[Cache]:
        return self._cache

    def _get_iterable(self, iterable: Any) -> Any:
        """
        Get an iterable or a tqdm progress bar.

        :param iterable: the iterable to be processed.
        :return: the iterable or the tqdm progress bar.
        """
        if self._show_progress and len(iterable) >= self._min_size_to_show_progress:
            return tqdm(iterable)
        else:
            return iterable

    def embed_query(self, query: str) -> Vector:
        """
        Embeds a query string.

        The implementation of this method delegates the embedding of the query
        string to the _embed_text() method.

        :param query: the query string to be embedded.
        :return: the embedded vector of the query string.
        """
        self._logger.info("Embedding a query: %s", query)
        vector = self._embed_text(query)
        self._logger.info("Successfully embedded the query.")
        self._logger.debug("The embedded vector of the query is: %s", vector)
        return vector

    def embed_document(self, document: Document) -> Point:
        """
        Embeds a document.

        The implementation of this method delegates the embedding of the content
        of the document to the _embed_text() method.

        :param document: the document to be embedded.
        :return: the embedded points of the document.
        """
        self._logger.info("Embedding a document ...")
        self._logger.debug("The document to be embedded is: %s", document)
        if not document.id:
            document.id = self._id_generator.generate()
        vector = self._embed_text(document.content)
        point = Point.from_document(document, vector)
        self._logger.info("Successfully embedded the document.")
        self._logger.debug("The embedded point of the document is: %s", point)
        return point

    def embed_documents(self, documents: List[Document]) -> List[Point]:
        """
        Embeds a list of documents.

        The implementation of this method delegates the embedding of contents of
        documents to the _embed_texts() method.

        :param documents: the list of documents.
        :return: the list of embedded points of each document.
        """
        n = len(documents)
        self._logger.info("Embedding %d documents ...", n)
        self._logger.debug("The documents to be embedded are: %s", documents)
        texts = [doc.content for doc in self._get_iterable(documents)]
        vectors = self._embed_texts(texts)
        points = []
        for i in self._get_iterable(range(n)):
            if not documents[i].id:
                documents[i].id = self._id_generator.generate()
            p = Point.from_document(documents[i], vectors[i])
            points.append(p)
        self._logger.info("Successfully embedded %d documents.", n)
        self._logger.debug("The embedded points of the documents are: %s", points)
        return points

    def embed_text(self, text: str) -> Vector:
        """
        Embeds a piece of text.

        The implementation of this method delegates the embedding of the text to
        the _embed_text() method.

        :param text: the specified text.
        :return: the embedded vector of the text.
        """
        self._logger.info("Embedding a piece of text: %s", text)
        vector = self._embed_text(text)
        self._logger.info("Successfully embedded the text.")
        self._logger.debug("The embedded vector of the text is: %s", vector)
        return vector

    def embed_texts(self, texts: List[str]) -> List[Vector]:
        """
        Embeds a list of texts.

        The implementation of this method delegates the embedding of the list of
         texts to the _embed_texts() method.

        :param texts: the list of texts.
        :return: the list of embedded vectors of each text.
        """
        self._logger.info("Embedding a list of texts ...")
        self._logger.debug("The texts to be embedded are: %s", texts)
        vectors = self._embed_texts(texts)
        self._logger.info("Successfully embedded the list of texts.")
        self._logger.debug("The embedded vectors of the texts are: %s", vectors)
        return vectors

    def _embed_text(self, text: str) -> Vector:
        """
        Embeds a piece of text.

        The implementation of this method delegates the embedding of the text
        to the _embed_impl() method. It checks the state of this embedding model,
        and take consideration of the cache.

        :param text: the specified text.
        :return: the embedded vector of the text.
        """
        if self._cache is None:
            self._logger.info("Embedding cache is disabled. "
                              "Embedding the text directly.")
            return self._embed_impl([text])[0]
        else:
            self._logger.info("Embedding cache is enabled.")
            if text in self._cache:
                self._logger.info("The text is found in the cache.")
                return self._cache[text]
            else:
                self._logger.info("The text is not found in the cache. "
                                  "Embedding it directly.")
                vector = self._embed_impl([text])[0]
                self._cache[text] = vector
                return vector

    def _embed_texts(self, texts: List[str]) -> List[Vector]:
        """
        Embeds a list of texts.

        The implementation of this method delegates the embedding of the texts
        to the _embed_impl() method. It checks the state of this embedding model,
        and take consideration of the cache.

        :param texts: the list of texts.
        :return: the list of embedded vectors of each text.
        """
        if self._cache is None:
            self._logger.info("Embedding cache is disabled. "
                              "Embedding the texts directly.")
            return self._embed_impl(texts)
        else:
            self._logger.info("Embedding cache is enabled.")
            vectors = []
            # use a dict to remove duplicated uncached texts
            # the `uncached` dict maps an uncached text to its embedded vector
            uncached = dict()
            for text in self._get_iterable(texts):
                if text in self._cache:
                    self._logger.info("The text is found in the cache: %s", text)
                    vector = self._cache[text]
                else:
                    if text not in uncached:
                        self._logger.info("The text is NOT found in the cache: %s", text)
                    uncached[text] = None
                    vector = None
                vectors.append(vector)
            if len(uncached) == 0:
                return vectors
            # delegate to _embed_impl() to embed the uncached texts
            uncached_texts = list(uncached.keys())
            uncached_vectors = self._embed_impl(uncached_texts)
            # fill the cache and the mapping
            for i in self._get_iterable(range(len(uncached_texts))):
                text = uncached_texts[i]
                vector = uncached_vectors[i]
                uncached[text] = vector
                self._cache[text] = vector
            # fill the result vector list
            for i in self._get_iterable(range(len(texts))):
                if vectors[i] is None:
                    text = texts[i]
                    # note that we cannot use self._cache to replace the `uncached`
                    # dict, since the vectors stored in self._cache may be evicted
                    # if the size of the cache exceeds the capacity.
                    vectors[i] = uncached[text]
            return vectors

    @abstractmethod
    def _embed_impl(self, texts: List[str]) -> List[Vector]:
        """
        Implementation of embedding a list of texts.

        This method must be implemented by the subclasses. The implementation
        do NOT have to check the state of this embedding model, and do NOT have
        to consider the cache.

        :param texts: the list of texts.
        :return: the list of embedded vectors of each text.
        """
