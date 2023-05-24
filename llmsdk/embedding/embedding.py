# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import logging
from abc import ABC, abstractmethod
from typing import List

from ..common import Document, Vector, Point


class Embedding(ABC):
    """
    The interface of sentence embedding models.
    """

    TEXT_ATTRIBUTE: str = "__text__"
    """The name of the metadata attribute storing the original text of a document."""

    def __init__(self,
                 output_dimensions: int) -> None:
        """
        Creates a Embedding object.

        :param output_dimensions: the number of dimension of the embedded vectors.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._output_dimensions = output_dimensions

    @property
    def output_dimensions(self) -> int:
        """
        Gets the number of dimensions of the embedded vectors.

        :return: the number of dimensions of the embedded vectors.
        """
        return self._output_dimensions

    def embed_query(self, query: str) -> Vector:
        """
        Embeds a query string.

        The subclass may override the default implementation of this method
        for optimization.

        :param query: the query string to be embedded.
        :return: the embedded vector of the query string.
        """
        vectors = self._embed_texts([query])
        return vectors[0]

    def embed_document(self, document: Document) -> Point:
        """
        Embeds a document.

        :param document: the document to be embedded.
        :return: the embedded points of the document.
        """
        vectors = self._embed_texts([document.content])
        metadata = {
            Embedding.TEXT_ATTRIBUTE: document.content,
        }
        if document.metadata is not None:
            metadata.update(document.metadata)
        return Point(vectors[0], id=document.id, metadata=metadata)

    def embed_documents(self, documents: List[Document]) -> List[Point]:
        """
        Embeds a list of documents.

        :param documents: the list of documents.
        :return: the list of embedded points of each document.
        """
        vectors = self._embed_texts([doc.content for doc in documents])
        n = len(documents)
        result = []
        for i in range(n):
            doc = documents[i]
            vector = vectors[i]
            metadata = {
                Embedding.TEXT_ATTRIBUTE: doc.content,
            }
            if doc.metadata is not None:
                metadata.update(doc.metadata)
            result.append(Point(vector, id=doc.id, metadata=metadata))
        return result

    @abstractmethod
    def _embed_texts(self, texts: List[str]) -> List[Vector]:
        """
        Embeds a list of texts.

        :param texts: the list of texts.
        :return: the list of embedded vectors of each text.
        """
