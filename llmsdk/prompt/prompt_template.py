# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .prompt import Prompt


@dataclass
class PromptTemplate(ABC):
    """
    The interface of prompt templates.

    Most of the time, this prompt send to a model is not hardcoded but is rather
    dynamically created based on a combination of user input, other non-static
    information (often coming from multiple sources), and a fixed template string.

    We call the object responsible for creating the Prompt a PromptTemplate.
    This object exposes a method for taking in input variables and returning a
    Prompt.

    The template of this class only support the Python's "f-format" syntax.
    """

    @abstractmethod
    def format(self, **kwargs: Any) -> Prompt:
        """
        Formats the prompt with the specified input variables.

        Example:

        .. code-block:: python

            prompt.format(variable1="foo")

        :param kwargs: Any arguments to be passed to the prompt template.
        :return: the formatted prompt.
        """