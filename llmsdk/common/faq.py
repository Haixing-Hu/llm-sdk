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
