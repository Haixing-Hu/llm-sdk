# ##############################################################################
#                                                                              #
#    Copyright (c) 2023.  Haixing Hu.                                          #
#                                                                              #
#    All rights reserved.                                                      #
#                                                                              #
# ##############################################################################
import logging
from abc import ABC, abstractmethod


class Embedding(ABC):
    """
    Interface for embedding models.
    """
    def __init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """
        Embeds a list of documents.

        :param documents: the contents of a list of documents.
        :return: the list of embedding vectors of each document.
        """

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """
        Embeds a query string.

        :param query: the query string to be embedded.
        :return: the embedding vectors of the query string.
        """
