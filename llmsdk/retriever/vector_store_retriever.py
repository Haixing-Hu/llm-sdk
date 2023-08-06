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
from ..vectorstore.collection_info import CollectionInfo
from ..vectorstore.vector_store import VectorStore
from ..embedding.embedding import Embedding
from ..splitter.text_splitter import TextSplitter
from ..util.common_utils import extract_argument
from .retriever import Retriever


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

    def _open(self, **kwargs: Any) -> None:
        store = self._vector_store
        store.open(**kwargs)
        try:
            if store.has_collection(self._collection_name):
                store.open_collection(self._collection_name)
            else:
                self._logger.warning("No collection '%s' in the vector store '%s'. "
                                     "It will be automatically created.",
                                     self._collection_name,
                                     self._vector_store.store_name)
                store.create_collection(
                    collection_name=self._collection_name,
                    vector_size=self._embedding.vector_dimension,
                    distance=Distance.COSINE
                )
                store.open_collection(self._collection_name)
            self._query_vector_cache = {}
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
        self._query_vector_cache = {}
        self._is_opened = False

    def _retrieve(self, query: str, **kwargs: Any) -> List[Document]:
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
        self._logger.info("Adding a document to the collection '%s' of %s ...",
                          self._collection_name, self._retriever_name)
        self._logger.debug("The document to add is: %s", document)
        self._ensure_opened()
        docs = self._splitter.split_document(document)
        self._logger.debug("The document is splitted into %d sub-documents: %s",
                           len(docs), docs)
        points = self._embedding.embed_documents(docs)
        self._vector_store.add_all(points)
        self._logger.info("Successfully added the document to the collection '%s' "
                          "of %s.", self._collection_name, self._retriever_name)
        return docs

    def add_all(self, documents: List[Document]) -> List[Document]:
        """
        Adds a list of documents to this retriever.

        :param documents: the list of documents to add.
        :return: the list of actual documents added to this retriever, which may
            be the sub-documents splitted from the original document.
        """
        self._logger.info("Adding %d documents to the collection '%s' of %s ...",
                          len(documents), self._collection_name,
                          self._retriever_name)
        self._logger.debug("The documents to add are: %s", documents)
        self._ensure_opened()
        docs = self._splitter.split_documents(documents)
        self._logger.debug("The document is splitted into %d sub-documents: %s",
                           len(docs), docs)
        points = self._embedding.embed_documents(docs)
        self._vector_store.add_all(points)
        self._logger.info("Successfully added all documents to the collection '%s' "
                          "of %s.", self._collection_name, self._retriever_name)
        return docs

    def get_store_info(self) -> CollectionInfo:
        """
        Gets the information of the collection of underlying vector store.

        :return: the information of the collection of underlying vector store.
        """
        return self._vector_store.get_collection_info(self._collection_name)
