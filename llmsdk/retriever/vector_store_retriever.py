# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import Any, List

from .retriever import Retriever
from ..common import Document, SearchType, Distance
from ..vectorstore import VectorStore, CollectionInfo
from ..embedding import Embedding
from ..splitter import TextSplitter
from ..util.common_utils import extract_argument


class VectorStoreRetriever(Retriever):
    """
    A retriever based on a vector store.
    """

    DEFAULT_LIMIT = 10

    def __init__(self,
                 vector_store: VectorStore,
                 collection_name: str,
                 embedding: Embedding,
                 splitter: TextSplitter,
                 search_type: SearchType = SearchType.SIMILARITY) -> None:
        """
        Creates a VectorStoreRetriever.

        :param vector_store: the underlying vector store.
        :param collection_name: name of the vector collection to use.
        :param embedding: the underlying embedding model.
        :param splitter: the text splitter used to split documents.
        :param search_type: the searching type.
        """
        super().__init__()
        self._vector_store = vector_store
        self._collection_name = collection_name
        self._embedding = embedding
        self._splitter = splitter
        self._search_type = search_type
        self._query_vector_cache = {}

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
    def search_type(self) -> SearchType:
        return self._search_type

    def open(self) -> None:
        """
        Opens this vector store retriever.

        This function will open the underlying vector store of this retriever, and
        open the specified collection in the vector store.
        """
        store = self._vector_store
        store.open()
        if store.has_collection(self._collection_name):
            store.open_collection(self._collection_name)
        else:
            store.create_collection(collection_name=self._collection_name,
                                    vector_size=self._embedding.output_dimensions,
                                    distance=Distance.COSINE)
            store.open_collection(self._collection_name)
        self._query_vector_cache = {}
        super().open()

    def close(self) -> None:
        """
        Closes this vector store retriever.

        This function will close the specified collection in the underlying vector
        store, and close the vector store.
        """
        store = self._vector_store
        store.close_collection()
        store.close()
        self._query_vector_cache = {}
        super().close()

    def retrieve(self, query: str, **kwargs: Any) -> List[Document]:
        self._ensure_opened()
        self._logger.debug("Retrieve with query: %s", query)
        if query in self._query_vector_cache:
            query_vector = self._query_vector_cache[query]
        else:
            query_vector = self._embedding.embed_query(query)
            self._query_vector_cache[query] = query_vector
        self._logger.debug("Query the vector store with: %s", query_vector)
        limit = extract_argument(kwargs, "limit", VectorStoreRetriever.DEFAULT_LIMIT)
        score_threshold = extract_argument(kwargs, "score_threshold", None)
        criterion = extract_argument(kwargs, "criterion", None)
        match self._search_type:
            case SearchType.SIMILARITY:
                points = self._vector_store.similarity_search(
                    query_vector=query_vector,
                    limit=limit,
                    score_threshold=score_threshold,
                    criterion=criterion,
                    **kwargs,
                )
            case SearchType.MAX_MARGINAL_RELEVANCE:
                points = self._vector_store.max_marginal_relevance_search(
                    query_vector=query_vector,
                    limit=limit,
                    score_threshold=score_threshold,
                    criterion=criterion,
                    **kwargs,
                )
            case _:
                raise ValueError(f"Unsupported searching type: {self._search_type}")
        self._logger.debug("Gets the query result: %s", points)
        return Document.from_points(points)

    def add(self, document: Document) -> List[Document]:
        """
        Adds a document to this retriever.

        :param document: the document to add.
        :return: the list of actual documents added to this retriever, which may
            be the sub-documents splitted from the original document.
        """
        self._ensure_opened()
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
        self._ensure_opened()
        docs = self._splitter.split_documents(documents)
        points = self._embedding.embed_documents(docs)
        self._vector_store.add_all(points)
        return docs

    def get_store_info(self) -> CollectionInfo:
        """
        Gets the information of the collection of underlying vector store.

        :return: the information of the collection of underlying vector store.
        """
        return self._vector_store.get_collection_info(self._collection_name)
