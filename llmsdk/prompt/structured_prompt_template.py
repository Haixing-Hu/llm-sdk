# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from abc import ABC
from dataclasses import dataclass, field
from typing import List

from ..common import Example
from .prompt_template import PromptTemplate


@dataclass
class StructuredPromptTemplate(PromptTemplate, ABC):
    """
    The interface of a structured prompt templates.
    """

    prompt_template: str = ""
    """
    The template of the prompt of the final input.

    The template of prompt may contain formatting placeholders.
    """

    instruction_template: str = ""
    """
    The template of the instruction of the prompt.

    The template of instruction may contain formatting placeholders.
    """

    examples: List[Example] = field(default_factory=list)
    """
    The list of examples.
    
    Note that the examples should not contain formatting placeholders.
    """
