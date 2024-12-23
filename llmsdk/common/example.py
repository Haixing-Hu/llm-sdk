# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

from .document import Document, ATTRIBUTE_DOCUMENT_TYPE
from .metadata import Metadata


ATTRIBUTE_EXAMPLE_ID: str = "__example_id__"
"""
The name of the metadata attribute storing the ID of an Example. 
"""

ATTRIBUTE_EXAMPLE_INPUT: str = "__example_input__"
"""
The name of the metadata attribute storing the original input of an Example.
"""

ATTRIBUTE_EXAMPLE_OUTPUT: str = "__example_output__"
"""
The name of the metadata attribute storing the original output of an Example.
"""

ATTRIBUTE_EXAMPLE_PART: str = "__example_part__"
"""
The name of the metadata attribute storing the name of the part of an Example.
"""


@dataclass(frozen=True)
class Example:
    """
    Examples are input/output pairs that represent inputs to a function and then
    expected output. They can be used in both training and evaluation of models.

    These can be inputs/outputs for a model or for a chain. Both types of
    examples serve a different purpose. Examples for a model can be used to
    finetune a model. Examples for a chain can be used to evaluate the
    end-to-end chain, or maybe even train a model to replace that whole chain.
    """

    input: str
    """The input of the example"""

    output: str
    """The output of the example"""

    id: str = None
    """The ID of the example."""

    score: Optional[float] = None
    """The score of this example relevant to the query."""

    def __eq__(self, other):
        """
        Tests whether this object is equal to another object.

        Two examples are equal if and only if all their fields except the score
        field are equal.

        :param other: the other object.
        :return; true if this object is equal to the other object; false otherwise.
        """
        if isinstance(other, Example):
            return (self.id == other.id
                    and self.input == other.input
                    and self.output == other.output)
        else:
            return False

    @classmethod
    def to_document(cls, example: Example) -> List[Document]:
        """
        Converts an example to a list of documents.

        :param example: the example.
        :return: the list of documents converted from the example.
        """
        if example.id is None or len(example.id) == 0:
            raise ValueError(f"The example must have a non-empty ID: {example}")
        input_doc = Document(id=example.id + "-input",
                             content=example.input,
                             metadata=Metadata({
                                 ATTRIBUTE_DOCUMENT_TYPE: "EXAMPLE",
                                 ATTRIBUTE_EXAMPLE_PART: "INPUT",
                                 ATTRIBUTE_EXAMPLE_ID: example.id,
                                 ATTRIBUTE_EXAMPLE_INPUT: example.input,
                                 ATTRIBUTE_EXAMPLE_OUTPUT: example.output,
                             }),
                             score=example.score)
        output_doc = Document(id=example.id + "-output",
                              content=example.output,
                              metadata=Metadata({
                                  ATTRIBUTE_DOCUMENT_TYPE: "EXAMPLE",
                                  ATTRIBUTE_EXAMPLE_PART: "OUTPUT",
                                  ATTRIBUTE_EXAMPLE_ID: example.id,
                                  ATTRIBUTE_EXAMPLE_INPUT: example.input,
                                  ATTRIBUTE_EXAMPLE_OUTPUT: example.output,
                              }),
                              score=example.score)
        return [input_doc, output_doc]

    @classmethod
    def to_documents(cls, examples: List[Example]) -> List[Document]:
        """
        Converts a list of examples to a list of documents.

        :param examples: the specified list of examples.
        :return: the list of documents converted from the specified list of
            examples.
        """
        result = []
        for example in examples:
            result.extend(Example.to_document(example))
        return result

    @classmethod
    def is_example(cls, doc: Document) -> bool:
        """
        Tests whether this document is converted from an example.

        :return: True if this document is converted from an example; False
            otherwise.
        """
        metadata = doc.metadata
        return (metadata is not None
                and metadata.has_value_of_type(ATTRIBUTE_DOCUMENT_TYPE, str)
                and metadata[ATTRIBUTE_DOCUMENT_TYPE] == "EXAMPLE"
                and metadata.has_value_of_type(ATTRIBUTE_EXAMPLE_ID, str)
                and metadata.has_value_of_type(ATTRIBUTE_EXAMPLE_INPUT, str)
                and metadata.has_value_of_type(ATTRIBUTE_EXAMPLE_OUTPUT, str)
                and metadata.has_value_of_type(ATTRIBUTE_EXAMPLE_PART, str)
                and (metadata[ATTRIBUTE_EXAMPLE_PART] == "INPUT"
                     or metadata[ATTRIBUTE_EXAMPLE_PART] == "OUTPUT"))

    @classmethod
    def from_document(cls, document: Document) -> Example:
        """
        Converts the specified document to an example.

        :param document: the specified document.
        :return: the example the specified document is converted from.
        :raise ValueError: if the specified document is not converted from an
            example.
        """
        if not cls.is_example(document):
            raise ValueError(f"The document is not converted from an example: {document}")
        return Example(id=document.metadata[ATTRIBUTE_EXAMPLE_ID],
                       input=document.metadata[ATTRIBUTE_EXAMPLE_INPUT],
                       output=document.metadata[ATTRIBUTE_EXAMPLE_OUTPUT],
                       score=document.score)

    @classmethod
    def from_documents(cls, documents: List[Document]) -> List[Example]:
        """
        Converts the specified list of documents to a list of examples.

        :param documents: the specified list of documents.
        :return: the list of examples the specified documents are converted from.
        :raise ValueError: if any document is not converted from an example.
        """
        return [Example.from_document(doc) for doc in documents]
