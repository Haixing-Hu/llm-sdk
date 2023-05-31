# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from dataclasses import dataclass
from typing import Optional

@dataclass
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
