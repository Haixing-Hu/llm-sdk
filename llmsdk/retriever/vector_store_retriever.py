# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import Any, List

from .retriever import Retriever
from ..common import Document
from ..vectorstore import VectorStore
from ..embedding import Embedding


class VectorStoreRetriever(Retriever):
    """
    A retriever based on a vector store.
    """

    def __init__(self,
                 vector_store: VectorStore,
                 embedding: Embedding,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._vector_store = vector_store
        self._embedding = embedding

    @property
    def vector_store(self) -> VectorStore:
        return self._vector_store

    @property
    def embedding(self) -> Embedding:
        return self._embedding

    def get_relevant_documents(self, query: str) -> List[Document]:
        pass
