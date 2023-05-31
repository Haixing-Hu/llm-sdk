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
