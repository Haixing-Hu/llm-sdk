# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, TypeVar

from llmsdk.common import Document

class VectorStore(ABC):
    """
    The interface of vector stores.
    """

    @abstractmethod
    def add_texts(self,
                  texts: Iterable[str],
                  metadata: Optional[List[dict]] = None,
                  **kwargs: Any) -> List[str]:
        """
        Run more texts through the embeddings and add to the vector store.

        :param texts: Iterable of strings to add to the vector store.
        :param metadata: Optional list of metadata associated with the texts.
        :param kwargs: vector store specific parameters.
        :return: List of IDs from adding the texts into the vector store.
        """

    def add_documents(self,
                      documents: List[Document],
                      **kwargs: Any) -> List[str]:
        """
        Run more texts through the embeddings and add to the vector store.

        :param documents: List of documents to add to the vector store.
        :param kwargs: vector store specific parameters.
        :return: List of IDs from adding the documents into the vector store.
        """
