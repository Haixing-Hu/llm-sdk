# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

from .example import Example
from .metadata import Metadata
from .document import Document, DOCUMENT_TYPE_ATTRIBUTE


FAQ_ID_ATTRIBUTE: str = "__faq_id__"
"""
The name of the metadata attribute storing the ID of a FAQ. 
"""

FAQ_QUESTION_ATTRIBUTE: str = "__faq_question__"
"""
The name of the metadata attribute storing the original question of a FAQ.
"""

FAQ_ANSWER_ATTRIBUTE: str = "__faq_answer__"
"""
The name of the metadata attribute storing the original answer of a FAQ.
"""

FAQ_PART_ATTRIBUTE: str = "__faq_property__"
"""
The name of the metadata attribute storing the name of the part of a FAQ.
"""


@dataclass(frozen=True)
class Faq:
    """
    The class of question/answer pairs.
    """

    question: str
    """The question."""

    answer: str
    """The answer."""

    id: str = None
    """The ID of this object."""

    score: Optional[float] = None
    """The score of this FAQ relevant to the query."""

    def __eq__(self, other):
        """
        Tests whether this object is equal to another object.

        Two FAQs are equal if and only if all their fields except the score
        field are equal.

        :param other: the other object.
        :return; true if this object is equal to the other object; false otherwise.
        """
        if isinstance(other, Faq):
            return (self.id == other.id
                    and self.question == other.question
                    and self.answer == other.answer)
        else:
            return False

    @classmethod
    def to_example(cls, faq: Faq) -> Example:
        """
        Converts a FAQ to an example.
        """
        return Example(id=faq.id,
                       input=faq.question,
                       output=faq.answer,
                       score=faq.score)

    @classmethod
    def to_examples(cls, faqs: List[Faq]) -> List[Example]:
        return [Faq.to_example(f) for f in faqs]

    @classmethod
    def from_example(cls, example: Example) -> Faq:
        """
        Converts an example to a FAQ.

        :param example: the example.
        :return: the converted FAQ.
        """
        return Faq(id=example.id,
                   question=example.input,
                   answer=example.output,
                   score=example.score)

    @classmethod
    def from_examples(cls, examples: List[Example]) -> List[Faq]:
        """
        Converts a list of examples to a list of FAQs.

        :param examples: the list of examples.
        :return: the list of converted FAQs.
        """
        return [Faq.from_example(e) for e in examples]

    @classmethod
    def to_document(cls, faq: Faq) -> List[Document]:
        """
        Converts a FAQ to a list of documents.

        :return: the list of documents converted from the FAQ.
        """
        if faq.id is None or len(faq.id) == 0:
            raise ValueError(f"The FAQ must have a non-empty ID: {faq}")
        question_doc = Document(id=faq.id + "-question",
                                content=faq.question,
                                metadata=Metadata({
                                    DOCUMENT_TYPE_ATTRIBUTE: "FAQ",
                                    FAQ_PART_ATTRIBUTE: "question",
                                    FAQ_ID_ATTRIBUTE: faq.id,
                                    FAQ_QUESTION_ATTRIBUTE: faq.question,
                                    FAQ_ANSWER_ATTRIBUTE: faq.answer,
                                }),
                                score=faq.score)
        answer_doc = Document(id=faq.id + "-answer",
                              content=faq.answer,
                              metadata=Metadata({
                                  DOCUMENT_TYPE_ATTRIBUTE: "FAQ",
                                  FAQ_PART_ATTRIBUTE: "answer",
                                  FAQ_ID_ATTRIBUTE: faq.id,
                                  FAQ_QUESTION_ATTRIBUTE: faq.question,
                                  FAQ_ANSWER_ATTRIBUTE: faq.answer,
                              }),
                              score=faq.score)
        return [question_doc, answer_doc]

    @classmethod
    def to_documents(cls, faqs: List[Faq]) -> List[Document]:
        """
        Converts a list of FAQs to a list of documents.

        :return: the list of documents converted from the FAQs.
        """
        return [d for f in faqs for d in Faq.to_document(f)]

    @classmethod
    def is_faq(cls, doc: Document) -> bool:
        """
        Tests whether a document is converted from a FAQ.
        :param doc: the document to be tested.
        :return: True if the document is converted from a FAQ; false otherwise.
        """
        return ((doc.metadata is not None)
                and doc.metadata.has_value_of_type(DOCUMENT_TYPE_ATTRIBUTE, str)
                and (doc.metadata[DOCUMENT_TYPE_ATTRIBUTE] == "FAQ")
                and doc.metadata.has_value_of_type(FAQ_ID_ATTRIBUTE, str)
                and doc.metadata.has_value_of_type(FAQ_PART_ATTRIBUTE, str)
                and doc.metadata.has_value_of_type(FAQ_QUESTION_ATTRIBUTE, str)
                and doc.metadata.has_value_of_type(FAQ_ANSWER_ATTRIBUTE, str)
                and (doc.metadata[FAQ_PART_ATTRIBUTE] == "question"
                     or doc.metadata[FAQ_PART_ATTRIBUTE] == "answer"))

    @classmethod
    def from_document(cls, doc: Document) -> Faq:
        """
        Converts a document to a FAQ.

        :param doc: the document.
        :return: the converted FAQ.
        """
        if not cls.is_faq(doc):
            raise ValueError(f"The document is not converted from a FAQ: {doc}")
        return Faq(id=doc.metadata[FAQ_ID_ATTRIBUTE],
                   question=doc.metadata[FAQ_QUESTION_ATTRIBUTE],
                   answer=doc.metadata[FAQ_ANSWER_ATTRIBUTE],
                   score=doc.score)

    @classmethod
    def from_documents(cls, docs: List[Document]) -> List[Faq]:
        """
        Converts a list of documents to a list of FAQs.

        :param docs: the list of documents.
        :return: the list of converted FAQs.
        """
        return [Faq.from_document(d) for d in docs]
