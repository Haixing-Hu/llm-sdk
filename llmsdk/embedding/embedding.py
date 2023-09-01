# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from abc import ABC, abstractmethod
from typing import List, Optional, Iterable

from ..common.document import Document
from ..common.point import Point
from ..common.vector import Vector
from ..mixin.with_progress_mixin import WithProgressMixin
from ..mixin.with_cache_mixin import WithCacheMixin
from ..mixin.with_logger_mixin import WithLoggerMixin
from ..generator.id_generator import IdGenerator
from ..generator.default_id_generator import DefaultIdGenerator
from ..util.common_utils import generate_uncached_unique_items


class Embedding(WithLoggerMixin, WithCacheMixin, WithProgressMixin, ABC):
    """
    The abstract base class of sentence embedding models.
    """

    def __init__(self, *,
                 vector_dimension: int,
                 id_generator: Optional[IdGenerator] = None,
                 **kwargs) -> None:
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
        :param show_progress_threshold: the minimum number of embedding texts
            to show the embedding progress.
        :param kwargs: the extra arguments passed to the constructor of the
            base class.
        """
        super().__init__(**kwargs)
        self._vector_dimension = vector_dimension
        self._id_generator = id_generator or DefaultIdGenerator()

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
        vector = self._embed_text(document.content)
        if not document.id:
            document.id = self._id_generator.generate()
        point = Point.from_document(document, vector)
        self._logger.info("Successfully embedded the document.")
        self._logger.debug("The embedded point of the document is: %s", point)
        return point

    def embed_documents(self, documents: Iterable[Document]) -> Iterable[Point]:
        """
        Embeds a list of documents.

        The implementation of this method delegates the embedding of contents of
        documents to the _embed_texts() method.

        Note that the implementation use generator to avoid the memory overflow
        when the number of documents is very large.

        :param documents: the list of documents.
        :return: the list of embedded points of each document.
        """
        self._logger.info("Embedding documents ...")
        texts = (doc.content for doc in documents)
        vectors = self._embed_texts(texts)
        points = (Point.from_document(doc, vector, self._id_generator)
                  for doc, vector in zip(documents, vectors))
        self._logger.info("Successfully embedded documents.")
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

    def embed_texts(self, texts: Iterable[str]) -> Iterable[Vector]:
        """
        Embeds a list of texts.

        The implementation of this method delegates the embedding of the list of
         texts to the _embed_texts() method.

        :param texts: the list of texts.
        :return: the list of embedded vectors of each text.
        """
        self._logger.info("Embedding a list of texts ...")
        vectors = self._embed_texts(texts)
        self._logger.info("Successfully embedded the list of texts.")
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
        if not self._use_cache:
            self._logger.info("Cache is disabled. Embedding the text directly.")
            vectors = list(self._embed_impl([text]))
            return vectors[0]
        else:
            self._logger.info("Embedding cache is enabled.")
            if text in self._cache:
                self._logger.info("The text is found in the cache.")
                return self._cache[text]
            else:
                self._logger.info("The text is not found in the cache. "
                                  "Embedding it directly.")
                vectors = list(self._embed_impl([text]))
                self._cache[text] = vectors[0]
                return vectors[0]

    def _embed_texts(self, texts: Iterable[str]) -> Iterable[Vector]:
        """
        Embeds a list of texts.

        The implementation of this method delegates the embedding of the texts
        to the _embed_impl() method. It checks the state of this embedding model,
        and take consideration of the cache.

        :param texts: the list of texts.
        :return: the list of embedded vectors of each text.
        """
        if not self._use_cache:
            self._logger.info("Cache is disabled. Embedding the texts directly.")
            return self._embed_impl(texts)
        else:
            self._logger.info("Embedding cache is enabled.")
            return self._embed_texts_with_cache(texts)

    def _embed_texts_with_cache(self, texts: Iterable[str]) -> Iterable[Vector]:
        unique_uncached_texts = generate_uncached_unique_items(texts, self._cache)
        vectors = self._embed_impl(unique_uncached_texts)
        vectors_iter = iter(vectors)
        for text in texts:
            if text in self._cache:
                yield self._cache[text]
            else:
                vector = next(vectors_iter)
                self._cache[text] = vector
                yield vector

    # def _embed_texts_with_cache(self, texts: Iterable[str]) -> Iterable[Vector]:
    #     vectors = []
    #     # use a dict to remove duplicated uncached texts
    #     # the `uncached` dict maps an uncached text to its embedded vector
    #     uncached = dict()
    #     self._logger.info("Checking cache for each text to be embedded...")
    #     for text in self._get_iterable(texts):
    #         if text in self._cache:
    #             vector = self._cache[text]
    #         else:
    #             uncached[text] = None
    #             vector = None
    #         vectors.append(vector)
    #     if len(uncached) == 0:
    #         return vectors
    #     # delegate to _embed_impl() to embed the uncached texts
    #     uncached_texts = list(uncached.keys())
    #     uncached_vectors = self._embed_impl(uncached_texts)
    #     self._logger.info("Filling the embedding cache...")
    #     # fill the cache and the mapping
    #     for i in self._get_iterable(range(len(uncached_texts))):
    #         text = uncached_texts[i]
    #         vector = uncached_vectors[i]
    #         uncached[text] = vector
    #         self._cache[text] = vector
    #     self._logger.info("Filling the embedded vector list...")
    #     # fill the result vector list
    #     # note that we cannot use self._cache to replace the `uncached`
    #     # dict, since the vectors stored in self._cache may be evicted
    #     # if the size of the cache exceeds the capacity.
    #     for i in self._get_iterable(range(len(texts))):
    #         if vectors[i] is None:
    #             text = texts[i]
    #             vectors[i] = uncached[text]
    #     return vectors

    @abstractmethod
    def _embed_impl(self, texts: Iterable[str]) -> Iterable[Vector]:
        """
        Implementation of embedding a list of texts.

        This method must be implemented by the subclasses. The implementation
        do NOT have to check the state of this embedding model, and do NOT have
        to consider the cache.

        :param texts: the texts to be embedded.
        :return: the embedded vectors of each text.
        """
