# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import Any, List

from .retriever import Retriever
from ..common import Document, SearchType
from ..vectorstore import VectorStore
from ..embedding import Embedding
from ..splitter import TextSplitter


class VectorStoreRetriever(Retriever):
    """
    A retriever based on a vector store.
    """

    DEFAULT_LIMIT = 10

    def __init__(self,
                 vector_store: VectorStore,
                 embedding: Embedding,
                 splitter: TextSplitter,
                 search_type: SearchType = SearchType.SIMILARITY,
                 **kwargs: Any) -> None:
        """
        Creates a VectorStoreRetriever.

        :param vector_store: the underlying vector store.
        :param embedding: the underlying embedding model.
        :param splitter: the text splitter used to split documents.
        :param search_type: the searching type.
        :param kwargs: the other arguments.
        """
        super().__init__(**kwargs)
        self._vector_store = vector_store
        self._embedding = embedding
        self._splitter = splitter
        self._search_type = search_type

    @property
    def vector_store(self) -> VectorStore:
        return self._vector_store

    @property
    def embedding(self) -> Embedding:
        return self._embedding

    @property
    def search_type(self) -> SearchType:
        return self._search_type

    def retrieve(self, query: str, **kwargs: Any) -> List[Document]:
        query_vector = self._embedding.embed_query(query)
        if "limit" in kwargs:
            limit = kwargs["limit"]
            kwargs.pop("limit")
        else:
            limit = VectorStoreRetriever.DEFAULT_LIMIT
        match self._search_type:
            case SearchType.SIMILARITY:
                points = self._vector_store.similarity_search(
                    query_vector=query_vector,
                    limit=limit,
                    **kwargs
                )
            case SearchType.MAX_MARGINAL_RELEVANCE:
                points = self._vector_store.max_marginal_relevance_search(
                    query_vector=query_vector,
                    limit=limit,
                    **kwargs
                )
            case _:
                raise ValueError(f"Unsupported searching type: {self._search_type}")
        return [p.to_document() for p in points]

    def add(self, document: Document) -> List[Document]:
        """
        Adds a document to this retriever.

        :param document: the document to add.
        :return: the list of actual documents added to this retriever, which may
            be the sub-documents splitted from the original document.
        """
        docs = self._splitter.split_document(document)
        points = self._embedding.embed_documents(docs)
        self._vector_store.add_all(points)
        return docs

    def add_all(self, documents: List[Document]) -> List[Document]:
        """
        Adds a list of documents to this retriever.

        :param documents: the list of documents to add.
        :return: the list of actual documents added to this retriever, which may
            be the sub-documents splitted from the original document.
        """
        docs = self._splitter.split_documents(documents)
        points = self._embedding.embed_documents(docs)
        self._vector_store.add_all(points)
        return docs
