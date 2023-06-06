# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from abc import ABC, abstractmethod
from typing import Any, List
import logging

from ..common import Document


class Retriever(ABC):
    """
    The interface of document retrievers.
    """

    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._is_opened = False

    @abstractmethod
    def retrieve(self,
                 query: str,
                 **kwargs: Any) -> List[Document]:
        """
        Retrieves documents relevant to a query.

        :param query: the specified query string.
        :param kwargs: other arguments.
        :return: the list of documents relevant to the query.
        """

    @property
    def is_opened(self) -> bool:
        """
        Tests whether the underlying vector store of this retriever is opened.

        :return: True if the underlying vector store of this retriever is opened;
            False otherwise.
        """
        return self._is_opened

    def open(self) -> None:
        """
        Opens this vector store retriever.

        This function will open the underlying vector store of this retriever, and
        open the specified collection in the vector store.
        """
        self._is_opened = True

    def close(self) -> None:
        """
        Closes this vector store retriever.

        This function will close the specified collection in the underlying vector
        store, and close the vector store.
        """
        self._is_opened = False

    def _ensure_opened(self) -> None:
        """
        Ensures that the retriever is opened.

        :raise ValueError: if the retriever is not opened yet.
        """
        if not self.is_opened:
            raise ValueError("The retriever is not opened yet.")
