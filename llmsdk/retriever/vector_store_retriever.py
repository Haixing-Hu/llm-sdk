# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from typing import Any, List

from ..common.search_type import SearchType
from ..common.distance import Distance
from ..common.document import Document
from ..common.point import Point
from ..mixin.with_progress_mixin import WithProgressMixin
from ..mixin.with_cache_mixin import WithCacheMixin
from ..vectorstore.collection_info import CollectionInfo
from ..vectorstore.vector_store import VectorStore
from ..embedding.embedding import Embedding
from ..splitter.text_splitter import TextSplitter
from ..util.common_utils import extract_argument
from .retriever import Retriever


class VectorStoreRetriever(WithProgressMixin, WithCacheMixin, Retriever):
    """
    A retriever based on a vector store.
    """

    DEFAULT_LIMIT = 10

    def __init__(self, *,
                 vector_store: VectorStore,
                 collection_name: str,
                 embedding: Embedding,
                 splitter: TextSplitter,
                 search_type: SearchType = SearchType.SIMILARITY,
                 **kwargs) -> None:
        """
        Creates a VectorStoreRetriever.

        :param vector_store: the underlying vector store.
        :param collection_name: name of the vector collection to use.
        :param embedding: the underlying embedding model.
        :param splitter: the text splitter used to split documents.
        :param search_type: the searching type.
        :param use_cache: indicates whether to use the cache to store the
            embedded vectors of texts. If this argument is True, the embedded
            vectors of texts will be cached in a LRU cache. Otherwise, the
            embedded vectors of texts will not be cached.
        :param cache_size: the number of text embeddings to be cached. This
            argument is ignored if the use_cache argument is False.
        :param show_progress: indicates whether to show the progress of
            splitting and embedding.
        :param show_progress_threshold: the minimum number of texts to show the
            splitting and embedding progress.
        :param kwargs: the extra arguments passed to the constructor of the
            base class.
        """
        super().__init__(**kwargs)
        self._collection_name = collection_name
        self._vector_store = vector_store
        self._embedding = embedding
        self._splitter = splitter
        self._search_type = search_type
        if 'show_progress' in kwargs:
            show_progress = kwargs['show_progress']
            self._vector_store.show_progress = show_progress
            self._embedding.show_progress = show_progress
            self._splitter.show_progress = show_progress
        if 'show_progress_threshold' in kwargs:
            show_progress_threshold = kwargs['show_progress_threshold']
            self._vector_store.show_progress_threshold = show_progress_threshold
            self._embedding.show_progress_threshold = show_progress_threshold
            self._splitter.show_progress_threshold = show_progress_threshold
        if 'use_cache' in kwargs:
            use_cache = kwargs['use_cache']
            if 'cache_size' in kwargs:
                cache_size = kwargs['cache_size']
                self._embedding.set_cache(use_cache, cache_size)
            else:
                self._embedding.set_cache(use_cache)

    @property
    def vector_store(self) -> VectorStore:
        return self._vector_store

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def embedding(self) -> Embedding:
        return self._embedding

    @property
    def splitter(self) -> TextSplitter:
        return self._splitter

    @property
    def search_type(self) -> SearchType:
        return self._search_type

    def set_cache(self,
                  use_cache: bool,
                  cache_size: int = WithCacheMixin.DEFAULT_CACHE_SIZE) -> None:
        """
        Sets the caching capacity of this object.

        :param use_cache: indicates whether to use the cache to store the
            embedded vectors of texts. If this argument is True, the embedded
            vectors of texts will be cached in a LRU cache. Otherwise, the
            embedded vectors of texts will not be cached.
        :param cache_size: the number of text embeddings to be cached. This
            argument is ignored if the use_cache argument is False.
        """
        super().set_cache(use_cache, cache_size)
        self._embedding.set_cache(use_cache, cache_size)

    @property
    def show_progress(self) -> bool:
        return super().show_progress

    @show_progress.setter
    def show_progress(self, value: bool) -> None:
        super().show_progress = value
        self._vector_store.show_progress = value
        self._embedding.show_progress = value
        self._splitter.show_progress = value

    @property
    def show_progress_threshold(self) -> int:
        return super().show_progress_threshold

    @show_progress_threshold.setter
    def show_progress_threshold(self, value: int) -> None:
        super().show_progress_threshold = value
        self._vector_store.show_progress_threshold = value
        self._embedding.show_progress_threshold = value
        self._splitter.show_progress_threshold = value

    def set_logging_level(self, level: int | str) -> None:
        """
        Sets the logging level of this object.

        :param level: the logging level to be set.
        """
        super().set_logging_level(level)
        self._vector_store.set_logging_level(level)
        self._embedding.set_logging_level(level)
        self._splitter.set_logging_level(level)

    def get_store_info(self) -> CollectionInfo:
        """
        Gets the information of the collection of underlying vector store.

        :return: the information of the collection of underlying vector store.
        """
        return self._vector_store.get_collection_info(self._collection_name)

    def _open(self, **kwargs: Any) -> None:
        store = self._vector_store
        store.open(**kwargs)
        try:
            if store.has_collection(self._collection_name):
                store.open_collection(self._collection_name)
            else:
                self._logger.info("No collection '%s' in the vector store '%s'. "
                                  "It will be automatically created.",
                                  self._collection_name,
                                  self._vector_store.store_name)
                store.create_collection(
                    collection_name=self._collection_name,
                    vector_size=self._embedding.vector_dimension,
                    distance=Distance.COSINE
                )
                store.open_collection(self._collection_name)
        except Exception:
            store.close()
            raise
        self._is_opened = True

    def _close(self) -> None:
        """
        Closes this vector store retriever.

        This function will close the specified collection in the underlying vector
        store, and close the vector store.
        """
        self._vector_store.close()
        self._is_opened = False

    def _retrieve(self, query: str, **kwargs: Any) -> List[Document]:
        query_vector = self._embedding.embed_query(query)
        self._logger.debug("Query the vector store with: %s", query_vector)
        limit = extract_argument(kwargs, "limit", VectorStoreRetriever.DEFAULT_LIMIT)
        score_threshold = extract_argument(kwargs, "score_threshold", None)
        criterion = extract_argument(kwargs, "criterion", None)
        points = self._vector_store.search(
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            criterion=criterion,
            search_type=self._search_type,
            **kwargs,
        )
        self._logger.debug("Gets the query result: %s", points)
        return Point.to_documents(points)

    def add(self, document: Document) -> List[Document]:
        """
        Adds a document to this retriever.

        :param document: the document to add.
        :return: the list of actual documents added to this retriever, which may
            be the sub-documents splitted from the original document.
        """
        self._logger.info("Adding a document to the collection '%s' of %s ...",
                          self._collection_name, self._name)
        self._logger.debug("The document to add is: %s", document)
        self._ensure_opened()
        docs = self._splitter.split_document(document)
        self._logger.debug("The document is splitted into %d sub-documents: %s",
                           len(docs), docs)
        points = self._embedding.embed_documents(docs)
        self._vector_store.add_all(points)
        self._logger.info("Successfully added the document to the collection '%s' "
                          "of %s.", self._collection_name, self._name)
        return docs

    def add_all(self, documents: List[Document]) -> List[Document]:
        """
        Adds a list of documents to this retriever.

        :param documents: the list of documents to add.
        :return: the list of actual documents added to this retriever, which may
            be the sub-documents splitted from the original document.
        """
        self._logger.info("Adding %d documents to the collection '%s' of %s ...",
                          len(documents), self._collection_name, self._name)
        self._logger.debug("The documents to add are: %s", documents)
        self._ensure_opened()
        docs = self._splitter.split_documents(documents)
        self._logger.debug("The document is splitted into %d sub-documents: %s",
                           len(docs), docs)
        points = self._embedding.embed_documents(docs)
        self._vector_store.add_all(points)
        self._logger.info("Successfully added all documents to the collection '%s' "
                          "of %s.", self._collection_name, self._name)
        return docs
