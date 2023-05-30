# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from abc import ABC, abstractmethod
from typing import List

from ..common import Document


class Retriever(ABC):
    """
    The interface of document retrievers.
    """

    @abstractmethod
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Gets documents relevant to a query.

        :param query: the specified query.
        :return: the list of documents relevant to the query.
        """
