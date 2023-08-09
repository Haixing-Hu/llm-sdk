# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..common.prompt import Prompt


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
    def format_prompt(self, **kwargs: Any) -> Prompt:
        """
        Formats the prompt with the specified input variables.

        Example:

        .. code-block:: python

            template.format_prompt(variable1="foo")

        :param kwargs: Any arguments to be passed to the prompt template.
        :return: the formatted prompt.
        """
