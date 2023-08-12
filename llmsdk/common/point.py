# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List
import copy

from .document import Document
from .metadata import Metadata
from .vector import Vector

DOCUMENT_ID_ATTRIBUTE: str = "__document_id__"
"""The name of the metadata attribute storing the ID of the document."""

DOCUMENT_CONTENT_ATTRIBUTE: str = "__document_content__"
"""The name of the metadata attribute storing the original content of the document."""


@dataclass
class Point:
    """
    The class of points.

    The points are the central entity that a VectorStore operates with. A point
    is a record consisting of a vector, an optional ID, an optional metadata,
    and an optional score.
    """

    vector: Vector = field(default_factory=list)
    """The list of coordinates of a vector"""

    metadata: Metadata = field(default_factory=Metadata)
    """The metadata of this point."""

    id: Optional[str] = None
    """The ID of this point."""

    score: Optional[float] = None
    """The score of this point, which is set for searching result."""

    def round_vector(self, digits: int) -> Point:
        """
        Rounds the coordinates of the vector of this point.

        :param digits: the number of digits to round.
        :return: this point itself.
        """
        self.vector = [round(x, digits) for x in self.vector]
        return self

    @classmethod
    def is_document(cls, point: Point) -> bool:
        """
        Tests whether a Point is converted from a Document.

        :param point: the specified point.
        :return: True if the specified point is converted from a document; False
            otherwise.
        """
        metadata = point.metadata
        return (metadata is not None
                and metadata.has_value_of_type(DOCUMENT_ID_ATTRIBUTE, str)
                and metadata.has_value_of_type(DOCUMENT_CONTENT_ATTRIBUTE, str))

    @classmethod
    def to_document(cls, point: Point) -> Document:
        """
        Convert a Point to a Document.

        :param point: the specified point. Note that the metadata of the point
            should have "__document_id__" and "__document_content__" attributes.
        :return: the Document constructed from the point.
        :raise ValueError: if the specified point is not converted from a document.
        """
        if not cls.is_document(point):
            raise ValueError(f"The point is not converted from a document: {point}")
        metadata = copy.deepcopy(point.metadata)
        id = metadata[DOCUMENT_ID_ATTRIBUTE]
        metadata.pop(DOCUMENT_ID_ATTRIBUTE)
        content = metadata[DOCUMENT_CONTENT_ATTRIBUTE]
        metadata.pop(DOCUMENT_CONTENT_ATTRIBUTE)
        return Document(id=id,
                        content=content,
                        metadata=metadata,
                        score=point.score)

    @classmethod
    def to_documents(cls, points: List[Point]) -> List[Document]:
        """
        Converts a list of points to a list of documents.

        :param points: the specified list of points.
        :return: the list of documents converted from the specified list of points.
        """
        return [Point.to_document(p) for p in points]

    @classmethod
    def from_document(cls, doc: Document, vector: Vector) -> Point:
        """
        Constructs a point from the specified document and its embedded vector.

        :param doc: the specified document.
        :param vector: the embedded vector of the content of the specified
            document.
        :return: the constructed Point.
        """
        if doc.id is None or len(doc.id) == 0:
            raise ValueError(f"The document must have a non-empty ID: {doc}")
        metadata = Metadata({
            DOCUMENT_ID_ATTRIBUTE: doc.id,
            DOCUMENT_CONTENT_ATTRIBUTE: doc.content,
        })
        if doc.metadata is not None:
            metadata.update(doc.metadata)
        # NOTE: should NOT set the ID of the point to the ID of the document,
        #   since the vector store may have its requirement on the format of
        #   the IDs of points.
        return Point(vector=vector, metadata=metadata, score=doc.score)

    @classmethod
    def from_documents(cls,
                       docs: List[Document],
                       vectors: List[Vector]) -> List[Point]:
        """
        Constructs a list of points from a list of documents and their embedded
        vectors.

        :param docs: the specified list of documents.
        :param vectors: the embedded vectors of the contents of the specified
            list of documents.
        :return: the constructed list of points.
        :raise ValueError: if the length of the list of documents does not equal
            the length of the list of vectors.
        """
        if len(docs) != len(vectors):
            raise ValueError("The length of the list of documents must equal to"
                             " the length of the list of vectors.")
        result = []
        for i, doc in enumerate(docs):
            point = Point.from_document(doc, vectors[i])
            result.append(point)
        return result
