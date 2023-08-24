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
from tqdm import tqdm

from ..common.search_type import SearchType
from ..vectorstore.collection_info import CollectionInfo
from ..vectorstore.vector_store import VectorStore
from ..embedding.embedding import Embedding
from ..llm.llm import LargeLanguageModel
from ..splitter.text_splitter import TextSplitter
from .retriever import Retriever
from .vector_store_retriever import VectorStoreRetriever


class VectorStoreBasedRetriever(Retriever, ABC):
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
                 use_cache: bool = True,
                 cache_size: int = 10000,
                 show_progress: bool = False,
                 min_size_to_show_progress: int = 10) -> None:
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
        :param show_progress: indicates whether to show the progress of adding
            records.
        :param min_size_to_show_progress: the minimum number of records to show
            the progress.
        """
        super().__init__()
        self._vector_store = vector_store
        self._collection_name = collection_name
        self._embedding = embedding
        self._splitter = splitter
        self._retriever = VectorStoreRetriever(
            vector_store=vector_store,
            collection_name=collection_name,
            embedding=embedding,
            splitter=splitter,
            search_type=search_type,
            use_cache=use_cache,
            cache_size=cache_size,
            show_progress=show_progress,
            min_size_to_show_progress=min_size_to_show_progress,
        )
        self._llm = llm
        self._use_cache = use_cache
        self._cache_size = cache_size
        self._show_progress = show_progress
        self._min_size_to_show_progress = min_size_to_show_progress

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
    def vector_store_retriever(self) -> VectorStoreRetriever:
        return self._retriever

    @property
    def large_language_model(self) -> LargeLanguageModel:
        return self._llm

    @property
    def use_cache(self) -> bool:
        return self._use_cache

    @property
    def cache_size(self) -> int:
        return self._cache_size

    def set_cache(self, use_cache: bool, cache_size: int) -> None:
        """
        Sets the caching capacity of this object.

        :param use_cache: indicates whether to use the cache to store the
            embedded vectors of texts. If this argument is True, the embedded
            vectors of texts will be cached in a LRU cache. Otherwise, the
            embedded vectors of texts will not be cached.
        :param cache_size: the number of text embeddings to be cached. This
            argument is ignored if the use_cache argument is False.
        """
        self._use_cache = use_cache
        self._cache_size = cache_size
        self._retriever.set_cache(use_cache, cache_size)

    @property
    def show_progress(self) -> bool:
        return self._show_progress

    @show_progress.setter
    def show_progress(self, value: bool) -> None:
        self._show_progress = value
        self._retriever.show_progress = value

    @property
    def min_size_to_show_progress(self) -> int:
        return self._min_size_to_show_progress

    @min_size_to_show_progress.setter
    def min_size_to_show_progress(self, value: int) -> None:
        self._min_size_to_show_progress = value
        self._retriever.min_size_to_show_progress = value

    def set_logging_level(self, level: int | str) -> None:
        """
        Sets the logging level of this retriever.

        :param level: the logging level to set.
        """
        self._logger.setLevel(level)
        self._retriever.set_logging_level(level)
        self._llm.set_logging_level(level)

    def get_store_info(self) -> CollectionInfo:
        """
        Gets the information of the collection of underlying vector store.

        :return: the information of the collection of underlying vector store.
        """
        return self._retriever.get_store_info()

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

    def _open(self, **kwargs: Any) -> None:
        self._retriever.open(**kwargs)
        self._is_opened = True

    def _close(self) -> None:
        self._retriever.close()
        self._is_opened = False
