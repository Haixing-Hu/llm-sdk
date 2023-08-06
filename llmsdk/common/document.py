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
from typing import List, ClassVar, Optional
import copy

from .metadata import Metadata
from .vector import Vector
from .point import Point

DOCUMENT_TYPE_ATTRIBUTE: str = "__type__"
"""
The name of the metadata attribute storing the type of a document. 
"""



@dataclass
class Document:
    """
    The model represents documents.

    A document has a title, a content, and optional metadata.
    """

    ORIGINAL_DOCUMENT_ID_ATTRIBUTE: ClassVar[str] = "__original_document_id__"

    ORIGINAL_DOCUMENT_INDEX_ATTRIBUTE: ClassVar[str] = "__original_document_index__"

    DOCUMENT_ID_ATTRIBUTE: ClassVar[str] = "__document_id__"
    """The name of the metadata attribute storing the ID of the document."""

    DOCUMENT_CONTENT_ATTRIBUTE: ClassVar[str] = "__document_content__"
    """The name of the metadata attribute storing the original text of the document."""

    content: str
    """The content of the document."""

    metadata: Metadata = field(default_factory=Metadata)
    """The metadata of the document, or {} if no metadata."""

    id: str = None
    """The ID of the document."""

    score: Optional[float] = None
    """The score of this document relevant to the query."""

    def is_splitted(self) -> bool:
        """
        Tests whether this document is a splitted document.
        :return: True if this document is a splitted document; False otherwise.
        """
        return (self.metadata.has_value_of_type(Document.ORIGINAL_DOCUMENT_ID_ATTRIBUTE, str)
                and self.metadata.has_value_of_type(Document.ORIGINAL_DOCUMENT_INDEX_ATTRIBUTE, int))

    def get_original_document_id(self) -> str:
        """
        Gets the ID of the original document from the metadata of this splitted
        document.

        :return: the ID of the original document where this specified document
            is splitted from.
        :raise ValueError: if there is any error while getting the ID of the
            original document from the metadata of this splitted document.
        """
        if not self.is_splitted():
            raise ValueError("This document is not a splitted document.")
        return self.metadata[Document.ORIGINAL_DOCUMENT_ID_ATTRIBUTE]

    def get_original_document_index(self) -> int:
        """
        Gets the index of this splitted document in its original document.

        :return: the index of this splitted document in its original document.
        :raise ValueError: if there is any error while getting the index of this
            splitted document in its original document.
        """
        if not self.is_splitted():
            raise ValueError("This document is not a splitted document.")
        return self.metadata[Document.ORIGINAL_DOCUMENT_INDEX_ATTRIBUTE]

    def get_original_document_metadata(self) -> Metadata:
        """
        Gets the metadata of the original document from the metadata of this
        splitted document.

        :return: the metadata of the original document.
        """
        if not self.is_splitted():
            raise ValueError("This document is not a splitted document.")
        result = copy.deepcopy(self.metadata)
        result.pop(Document.ORIGINAL_DOCUMENT_ID_ATTRIBUTE)
        result.pop(Document.ORIGINAL_DOCUMENT_INDEX_ATTRIBUTE)
        return result

    def create_splitted_document(self,
                                 text: str,
                                 index: int) -> Document:
        """
        Creates a splitted document from this original document.

        :param text: the splitted text of the content of this original document.
        :param index: the index of the splitted document.
        :return: the specified splitted document of the original document.
        """
        if (self.id is None) or (len(self.id) == 0):
            raise ValueError(f"The ID of the original document must be set: {self}")
        id = self.id + "-" + str(index)
        metadata = copy.deepcopy(self.metadata)
        metadata[Document.ORIGINAL_DOCUMENT_ID_ATTRIBUTE] = self.id
        metadata[Document.ORIGINAL_DOCUMENT_INDEX_ATTRIBUTE] = index
        return Document(id=id, content=text, metadata=metadata)

    @classmethod
    def from_point(cls, point: Point) -> Document:
        """
        Convert a Point to a Document.

        :param point: the specified point.
        :return: the Document constructed from the point. Note that the metadata
            of the point should have "__document_id__" and "__document_content__"
            attributes.
        :raise ValueError: if the specified point is not converted from a document.
        """
        if ((not point.metadata.has_value_of_type(Document.DOCUMENT_ID_ATTRIBUTE, str))
                or (not point.metadata.has_value_of_type(Document.DOCUMENT_CONTENT_ATTRIBUTE, str))):
            raise ValueError(f"The point is not converted from a document: {point}")
        id = point.metadata[Document.DOCUMENT_ID_ATTRIBUTE]
        content = point.metadata[Document.DOCUMENT_CONTENT_ATTRIBUTE]
        metadata = copy.deepcopy(point.metadata)
        metadata.pop(Document.DOCUMENT_ID_ATTRIBUTE)
        metadata.pop(Document.DOCUMENT_CONTENT_ATTRIBUTE)
        return Document(id=id, content=content, metadata=metadata, score=point.score)

    @classmethod
    def from_points(cls, points: List[Point]) -> List[Document]:
        """
        Converts a list of points to a list of documents.

        :param points: the specified list of points.
        :return: the list of documents converted from the specified list of points.
        """
        return [Document.from_point(p) for p in points]

    @classmethod
    def to_point(cls, document: Document, vector: Vector) -> Point:
        """
        Constructs a point from the specified document and its embedded vector.

        :param document: the specified document.
        :param vector: the embedded vector of the content of the specified
            document.
        :return: the constructed Point.
        """
        if document.id is None or len(document.id) == 0:
            raise ValueError(f"The document must have a non-empty ID: {document}")
        metadata = Metadata({
            Document.DOCUMENT_ID_ATTRIBUTE: document.id,
            Document.DOCUMENT_CONTENT_ATTRIBUTE: document.content,
        })
        if document.metadata is not None:
            metadata.update(document.metadata)
        # NOTE: should NOT set the ID of the point to the ID of the document,
        #   since the vector store may have its requirement on the format of
        #   the IDs of points.
        return Point(vector=vector, metadata=metadata, score=document.score)

    @classmethod
    def to_points(cls, documents: List[Document], vectors: List[Vector]) -> List[Point]:
        """
        Constructs a list of points from a list of documents and their embedded
        vectors.

        :param documents: the specified list of documents.
        :param vectors: the embedded vectors of the contents of the specified
            list of documents.
        :return: the constructed list of points.
        :raise ValueError: if the length of the list of documents does not equal
            the length of the list of vectors.
        """
        if len(documents) != len(vectors):
            raise ValueError("The length of the list of documents must equal to "
                             "the length of the list of vectors.")
        result = []
        for i, doc in enumerate(documents):
            point = Document.to_point(doc, vectors[i])
            result.append(point)
        return result
