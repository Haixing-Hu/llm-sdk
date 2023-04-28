# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import logging
from abc import ABC, abstractmethod
from typing import List

from llmsdk.common import Point, Document, Vector


class Embedding(ABC):
    """
    Interface for embedding models.
    """
    def __init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)

    def embed_query(self, query: str) -> Vector:
        """
        Embeds a query string.

        The subclass may override the default implementation of this method
        for optimization.

        :param query: the query string to be embedded.
        :return: the embedded vector of the query string.
        """
        document = Document(query, metadata={"type": "query"})
        vectors = self.embed_documents([document])
        return vectors[0]

    def embed_documents(self, documents: List[Document]) -> List[Vector]:
        """
        Embeds a list of documents.

        :param documents: the list of documents.
        :return: the list of embedded vectors of each document.
        """
        points = self._embed_texts([doc.content for doc in documents])
        n = len(documents)
        result = []
        for i in range(n):
            doc = documents[i]
            point = points[i]
            metadata = {"_text": doc.content}
            if doc.metadata is not None:
                metadata.update(doc.metadata)
            result.append(Vector(point, metadata=metadata))
        return result

    @abstractmethod
    def _embed_texts(self, texts: List[str]) -> List[Point]:
        """
        Embeds a list of texts.

        :param texts: the list of texts.
        :return: the list of embedded vectors of each text.
        """
