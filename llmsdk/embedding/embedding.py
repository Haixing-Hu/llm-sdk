# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from ..common.document import Document
from ..common.vector import Vector
from ..common.point import Point
from ..generator.id_generator import IdGenerator
from ..generator.default_id_generator import DefaultIdGenerator


class Embedding(ABC):
    """
    The interface of sentence embedding models.
    """

    def __init__(self,
                 vector_dimension: int,
                 id_generator: Optional[IdGenerator] = None) -> None:
        """
        Creates a Embedding object.

        :param vector_dimension: the number of dimension of the embedded vectors.
        :param id_generator: the generator used to generating document IDs.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._vector_dimension = vector_dimension
        self._id_generator = id_generator or DefaultIdGenerator()

    @property
    def vector_dimension(self) -> int:
        """
        Gets the number of dimensions of the embedded vectors.

        :return: the number of dimensions of the embedded vectors.
        """
        return self._vector_dimension

    @property
    def id_generator(self) -> IdGenerator:
        return self._id_generator

    def embed_query(self, query: str) -> Vector:
        """
        Embeds a query string.

        The subclass may override the default implementation of this method
        for optimization.

        :param query: the query string to be embedded.
        :return: the embedded vector of the query string.
        """
        self._logger.info("Embedding a query: %s", query)
        vectors = self._embed_texts([query])
        self._logger.info("Successfully embedded the query.")
        self._logger.debug("The embedded vector of the query is: %s", vectors[0])
        return vectors[0]

    def embed_document(self, document: Document) -> Point:
        """
        Embeds a document.

        :param document: the document to be embedded.
        :return: the embedded points of the document.
        """
        self._logger.info("Embedding a document ...")
        self._logger.debug("The document to be embedded is: %s", document)
        if not document.id:
            document.id = self._id_generator.generate()
        vectors = self._embed_texts([document.content])
        point = Document.to_point(document, vectors[0])
        self._logger.info("Successfully embedded the document.")
        self._logger.debug("The embedded point of the document is: %s", point)
        return point

    def embed_documents(self, documents: List[Document]) -> List[Point]:
        """
        Embeds a list of documents.

        :param documents: the list of documents.
        :return: the list of embedded points of each document.
        """
        n = len(documents)
        self._logger.info("Embedding %d documents ...", n)
        self._logger.debug("The documents to be embedded are: %s", documents)
        vectors = self._embed_texts([doc.content for doc in documents])
        points = []
        for i in range(n):
            if not documents[i].id:
                documents[i].id = self._id_generator.generate()
            p = Document.to_point(documents[i], vectors[i])
            points.append(p)
        self._logger.info("Successfully embedded %d documents.", n)
        self._logger.debug("The embedded points of the documents are: %s", points)
        return points

    def embed_texts(self, texts: List[str]) -> List[Vector]:
        """
        Embeds a list of texts.

        :param texts: the list of texts.
        :return: the list of embedded vectors of each text.
        """
        self._logger.info("Embedding a list of texts ...")
        self._logger.debug("The texts to be embedded are: %s", texts)
        vectors = self._embed_texts(texts)
        self._logger.info("Successfully embedded the list of texts.")
        self._logger.debug("The embedded vectors of the texts are: %s", vectors)
        return vectors

    @abstractmethod
    def _embed_texts(self, texts: List[str]) -> List[Vector]:
        """
        Embeds a list of texts.

        This method should be implemented by the subclasses. The implementation
        do not have to check the state of this embedding model.

        :param texts: the list of texts.
        :return: the list of embedded vectors of each text.
        """

    def embed_text(self, text: str) -> Vector:
        """
        Embeds a piece of text.

        :param text: the specified text.
        :return: the embedded vector of the text.
        """
        self._logger.info("Embedding a piece of text: %s", text)
        vectors = self._embed_texts([text])
        self._logger.info("Successfully embedded the text.")
        self._logger.debug("The embedded vector of the text is: %s", vectors[0])
        return vectors[0]
