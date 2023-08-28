# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from abc import ABC
from typing import Any

from ..common.search_type import SearchType
from ..mixin.with_progress_mixin import WithProgressMixin
from ..mixin.with_cache_mixin import WithCacheMixin
from ..vectorstore.collection_info import CollectionInfo
from ..vectorstore.vector_store import VectorStore
from ..embedding.embedding import Embedding
from ..llm.llm import LargeLanguageModel
from ..splitter.text_splitter import TextSplitter
from .retriever import Retriever
from .vector_store_retriever import VectorStoreRetriever


class VectorStoreBasedRetriever(WithProgressMixin, WithCacheMixin, Retriever, ABC):
    """
    The abstract base class of retrievers that based on a VectorStoreRetriever
    and a LargeLanguageModel.
    """
    def __init__(self, *,
                 vector_store: VectorStore,
                 collection_name: str,
                 embedding: Embedding,
                 splitter: TextSplitter,
                 llm: LargeLanguageModel,
                 search_type: SearchType = SearchType.SIMILARITY,
                 **kwargs) -> None:
        """
        Constructs a `VectorStoreBasedRetriever`.

        :param vector_store: the underlying vector store.
        :param collection_name: the name of the vector collection to use. The
            collection must store the vectors of the list of known records.
        :param embedding: the underlying embedding model.
        :param splitter: the text splitter used to split documents.
        :param llm: the underlying large language model.
        :param search_type: the searching type of the underlying vector store
            retriever.
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
        self._llm = llm
        self._retriever = VectorStoreRetriever(
            collection_name=collection_name,
            vector_store=vector_store,
            embedding=embedding,
            splitter=splitter,
            search_type=search_type,
            **kwargs,
        )

    @property
    def vector_store(self) -> VectorStore:
        return self._retriever.vector_store

    @property
    def collection_name(self) -> str:
        return self._retriever.collection_name

    @property
    def embedding(self) -> Embedding:
        return self._retriever.embedding

    @property
    def splitter(self) -> TextSplitter:
        return self._retriever.splitter

    @property
    def vector_store_retriever(self) -> VectorStoreRetriever:
        return self._retriever

    @property
    def large_language_model(self) -> LargeLanguageModel:
        return self._llm

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
        self._retriever.set_cache(use_cache, cache_size)

    @property
    def show_progress(self) -> bool:
        return super().show_progress

    @show_progress.setter
    def show_progress(self, value: bool) -> None:
        super().show_progress = value
        self._retriever.show_progress = value

    @property
    def show_progress_threshold(self) -> int:
        return super().show_progress_threshold

    @show_progress_threshold.setter
    def show_progress_threshold(self, value: int) -> None:
        super().show_progress_threshold = value
        self._retriever.show_progress_threshold = value

    def set_logging_level(self, level: int | str) -> None:
        """
        Sets the logging level of this retriever.

        :param level: the logging level to set.
        """
        super().set_logging_level(level)
        self._llm.set_logging_level(level)
        self._retriever.set_logging_level(level)

    def get_store_info(self) -> CollectionInfo:
        """
        Gets the information of the collection of underlying vector store.

        :return: the information of the collection of underlying vector store.
        """
        return self._retriever.get_store_info()

    def _open(self, **kwargs: Any) -> None:
        self._retriever.open(**kwargs)
        self._is_opened = True

    def _close(self) -> None:
        self._retriever.close()
        self._is_opened = False
