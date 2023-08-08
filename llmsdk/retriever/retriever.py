# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from abc import ABC, abstractmethod
from typing import Any, List
from logging import Logger, getLogger

from ..common.document import Document


class Retriever(ABC):
    """
    The interface of document retrievers.
    """

    def __init__(self):
        self._logger = getLogger(self.__class__.__name__)
        self._retriever_name = self.__class__.__name__
        self._is_opened = False

    @property
    def logger(self) -> Logger:
        return self._logger

    def set_logging_level(self, level: int | str) -> None:
        """
        Sets the logging level of this object.

        :param level: the logging level to be set.
        """
        self._logger.setLevel(level)

    @property
    def retriever_name(self) -> str:
        return self._retriever_name

    @property
    def is_opened(self) -> bool:
        """
        Tests whether the underlying vector store of this retriever is opened.

        :return: True if the underlying vector store of this retriever is opened;
            False otherwise.
        """
        return self._is_opened

    def open(self, **kwargs: Any) -> None:
        """
        Opens this vector store retriever.

        This function will open the underlying vector store of this retriever, and
        open the specified collection in the vector store.

        :param kwargs: the arguments used to open this retriever.
        """
        self._logger.info("Opening the %s...", self._retriever_name)
        self._ensure_closed()
        self._open(**kwargs)
        self._logger.info("Successfully opened the %s.", self._retriever_name)
        self._is_opened = True

    @abstractmethod
    def _open(self, **kwargs: Any) -> None:
        """
        Opens this vector store retriever.

        This function will open the underlying vector store of this retriever, and
        open the specified collection in the vector store.

        This method should be implemented by the subclasses. The implementation
        do not have to check the state of this retriever.

        :param kwargs: the arguments used to open this retriever.
        """

    def close(self) -> None:
        """
        Closes this vector store retriever.

        This function will close the specified collection in the underlying vector
        store, and close the vector store.
        """
        if self.is_opened:
            self._logger.info("Closing the %s...", self._retriever_name)
            self._close()
            self._logger.info("Successfully closed the %s.", self._retriever_name)

    @abstractmethod
    def _close(self) -> None:
        """
        Closes this vector store retriever.

        This function will close the specified collection in the underlying vector
        store, and close the vector store.

        This method should be implemented by the subclasses. The implementation
        do not have to check the state of this retriever.
        """

    def retrieve(self,
                 query: str,
                 **kwargs: Any) -> List[Document]:
        """
        Retrieves documents relevant to a query.

        :param query: the specified query string.
        :param kwargs: other arguments.
        :return: the list of documents relevant to the query.
        """
        self._logger.info("Retrieving documents from '%s' with query: %s",
                          self._retriever_name, query)
        if len(kwargs) > 0:
            self._logger.debug("kwargs = %s", kwargs)
        self._ensure_opened()
        result = self._retrieve(query, **kwargs)
        self._logger.info("Successfully retrieve documents from '%s'.",
                          self._retriever_name)
        self._logger.debug("The retrieved documents are: %s", result)
        return result

    @abstractmethod
    def _retrieve(self,
                  query: str,
                  **kwargs: Any) -> List[Document]:
        """
        Retrieves documents relevant to a query.

        This method should be implemented by the subclasses. The implementation
        do not have to check the state of this retriever.

        :param query: the specified query string.
        :param kwargs: other arguments.
        :return: the list of documents relevant to the query.
        """

    def _ensure_opened(self) -> None:
        """
        Ensures that the retriever is opened.

        :raise ValueError: if the retriever is not opened yet.
        """
        if not self.is_opened:
            raise ValueError("The retriever is not opened.")

    def _ensure_closed(self) -> None:
        """
        Ensures that the retriever is closed.

        :raise ValueError: if the retriever is not closed yet.
        """
        if self.is_opened:
            raise ValueError("The retriever is not closed.")
