# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

from .vector import Vector
from .metadata import Metadata
from .document import Document

DOCUMENT_ID_ATTRIBUTE: str = "__document_id__"
"""The name of the metadata attribute storing the ID of the document."""

DOCUMENT_CONTENT_ATTRIBUTE: str = "__document_content__"
"""The name of the metadata attribute storing the original text of the document."""


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

    metadata: Optional[Metadata] = field(default_factory=dict)
    """The metadata of this point."""

    id: Optional[str] = None
    """The ID of this point."""

    score: Optional[float] = None
    """The score of this point, which is set for searching result."""

    @classmethod
    def from_document(cls, document: Document, vector: Vector) -> Point:
        """
        Constructs a Point from a document and its embedded vector.

        :param document: the specified document.
        :param vector: the embedded vector of the content of the specified
            document.
        :return: the constructed Point.
        """
        metadata = {
            DOCUMENT_ID_ATTRIBUTE: document.id,
            DOCUMENT_CONTENT_ATTRIBUTE: document.content,
        }
        if document.metadata is not None:
            metadata.update(document.metadata)
        return Point(id=document.id, vector=vector, metadata=metadata)

    def to_document(self) -> Document:
        """
        Convert a Point to a Document.

        :return: the Document constructed from this point. Note that the metadata
            of this point should have "__doc_id__" and "__doc_content__" attributes.
        :raise ValueError: if the metadata of this point haven't "__doc_id__"
            and "__doc_content__" attributes.
        """
        if self.metadata is None:
            raise ValueError(f"No metadata in the point: {self}")
        if DOCUMENT_ID_ATTRIBUTE not in self.metadata:
            raise ValueError(f"No {DOCUMENT_ID_ATTRIBUTE} attribute in the "
                             f"metadata of the point: {self}")
        if DOCUMENT_CONTENT_ATTRIBUTE not in self.metadata:
            raise ValueError(f"No {DOCUMENT_CONTENT_ATTRIBUTE} attribute in the "
                             f"metadata of the point: {self}")
        id = self.metadata.get(DOCUMENT_ID_ATTRIBUTE)
        content = self.metadata.get(DOCUMENT_CONTENT_ATTRIBUTE)
        metadata = copy.deepcopy(self.metadata)
        metadata.pop(DOCUMENT_ID_ATTRIBUTE)
        metadata.pop(DOCUMENT_CONTENT_ATTRIBUTE)
        return Document(id=id, content=content, metadata=metadata)
