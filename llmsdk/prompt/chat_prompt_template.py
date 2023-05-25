# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from dataclasses import dataclass
from typing import Any

from .prompt_template import PromptTemplate


@dataclass
class ChatPromptTemplate(PromptTemplate):
    """
    The class of prompt templates which format intput variables into a list of
    chatting messages.
    """

    template: str
    """
    The template string.
    """

    def format(self, **kwargs: Any) -> str:
        return self.template.format(**kwargs)
