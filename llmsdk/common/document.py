# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, ClassVar
import copy

from .metadata import Metadata
from .example import Example
from .vector import Vector
from .point import Point


@dataclass
class Document:
    """
    The model represents documents.

    A document has a title, a content, and optional metadata.
    """

    ORIGINAL_DOCUMENT_ID_ATTRIBUTE: ClassVar[str] = "__original_document_id__"

    ORIGINAL_DOCUMENT_INDEX_ATTRIBUTE: ClassVar[str] = "__original_document_index__"

    EXAMPLE_ID_ATTRIBUTE: ClassVar[str] = "__example_id__"

    EXAMPLE_INPUT_ATTRIBUTE: ClassVar[str] = "__example_input__"

    EXAMPLE_OUTPUT_ATTRIBUTE: ClassVar[str] = "__example_output__"

    EXAMPLE_PROPERTY_ATTRIBUTE: ClassVar[str] = "__example_property__"

    DOCUMENT_ID_ATTRIBUTE: ClassVar[str] = "__document_id__"
    """The name of the metadata attribute storing the ID of the document."""

    DOCUMENT_CONTENT_ATTRIBUTE: ClassVar[str] = "__document_content__"
    """The name of the metadata attribute storing the original text of the document."""

    content: str
    """The content of the document."""

    id: str = None
    """The ID of the document."""

    metadata: Metadata = field(default_factory=Metadata)
    """The metadata of the document, or {} if no metadata."""

    def is_splitted(self) -> bool:
        """
        Tests whether this document is a splitted document.
        :return: True if this document is a splitted document; False otherwise.
        """
        return (self.metadata.has(Document.ORIGINAL_DOCUMENT_ID_ATTRIBUTE, str)
                and self.metadata.has(Document.ORIGINAL_DOCUMENT_INDEX_ATTRIBUTE, int))

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
    def from_example(cls, example: Example) -> List[Document]:
        """
        Converts an example to a list of documents.

        :return: the list of documents converted from the example.
        """
        if example.id is None or len(example.id) == 0:
            raise ValueError(f"The example must have a non-empty ID: {example}")
        input_doc = Document(id=example.id + "-input",
                             content=example.input,
                             metadata=Metadata({
                                 Document.EXAMPLE_ID_ATTRIBUTE: example.id,
                                 Document.EXAMPLE_PROPERTY_ATTRIBUTE: "input",
                                 Document.EXAMPLE_INPUT_ATTRIBUTE: example.input,
                                 Document.EXAMPLE_OUTPUT_ATTRIBUTE: example.output,
                             }))
        output_doc = Document(id=example.id + "-output",
                              content=example.output,
                              metadata=Metadata({
                                  Document.EXAMPLE_ID_ATTRIBUTE: example.id,
                                  Document.EXAMPLE_PROPERTY_ATTRIBUTE: "output",
                                  Document.EXAMPLE_INPUT_ATTRIBUTE: example.input,
                                  Document.EXAMPLE_OUTPUT_ATTRIBUTE: example.output,
                              }))
        return [input_doc, output_doc]

    @classmethod
    def from_examples(cls, examples: List[Example]) -> List[Document]:
        """
        Converts a list of examples to a list of documents.

        :param examples: the specified list of examples.
        :return: the list of documents converted from the specified list of
            examples.
        """
        result = []
        for example in examples:
            result.extend(Document.from_example(example))
        return result

    def is_converted_from_example(self) -> bool:
        """
        Tests whether this document is converted from an example.

        :return: True if this document is converted from an example; False
            otherwise.
        """
        return (self.metadata.has(Document.EXAMPLE_ID_ATTRIBUTE, str)
                and self.metadata.has(Document.EXAMPLE_PROPERTY_ATTRIBUTE, str)
                and self.metadata.has(Document.EXAMPLE_INPUT_ATTRIBUTE, str)
                and self.metadata.has(Document.EXAMPLE_OUTPUT_ATTRIBUTE, str)
                and (self.metadata[Document.EXAMPLE_PROPERTY_ATTRIBUTE] != "input"
                     or self.metadata[Document.EXAMPLE_PROPERTY_ATTRIBUTE] != "output"))

    def to_example(self) -> Example:
        """
        Converts this document to an example.

        :return: the example this document is converted from.
        :raise ValueError: if this document is not converted from an example.
        """
        if not self.is_converted_from_example():
            raise ValueError(f"The document is not converted from an example: {self}")
        example_id = self.metadata[Document.EXAMPLE_ID_ATTRIBUTE]
        example_input = self.metadata[Document.EXAMPLE_INPUT_ATTRIBUTE]
        example_output = self.metadata[Document.EXAMPLE_OUTPUT_ATTRIBUTE]
        return Example(id=example_id, input=example_input, output=example_output)

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
        if ((not point.metadata.has(Document.DOCUMENT_ID_ATTRIBUTE, str))
                or (not point.metadata.has(Document.DOCUMENT_CONTENT_ATTRIBUTE, str))):
            raise ValueError(f"The point is not converted from a document: {point}")
        id = point.metadata[Document.DOCUMENT_ID_ATTRIBUTE]
        content = point.metadata[Document.DOCUMENT_CONTENT_ATTRIBUTE]
        metadata = copy.deepcopy(point.metadata)
        metadata.pop(Document.DOCUMENT_ID_ATTRIBUTE)
        metadata.pop(Document.DOCUMENT_CONTENT_ATTRIBUTE)
        return Document(id=id, content=content, metadata=metadata)

    @classmethod
    def from_points(cls, points: List[Point]) -> List[Document]:
        """
        Converts a list of points to a list of documents.

        :param points: the specified list of points.
        :return: the list of documents converted from the specified list of points.
        """
        return [Document.from_point(p) for p in points]

    def to_point(self, vector: Vector) -> Point:
        """
        Constructs a Point from a document and its embedded vector.

        :param vector: the embedded vector of the content of the specified
            document.
        :return: the constructed Point.
        """
        if self.id is None or len(self.id) == 0:
            raise ValueError(f"The document must have a non-empty ID: {self}")
        metadata = Metadata({
            Document.DOCUMENT_ID_ATTRIBUTE: self.id,
            Document.DOCUMENT_CONTENT_ATTRIBUTE: self.content,
        })
        if self.metadata is not None:
            metadata.update(self.metadata)
        return Point(vector=vector, metadata=metadata)
