# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from abc import ABC, abstractmethod
from typing import Any, Optional

from .prompt import Prompt


class ReplyParser(ABC):
    """
    The interface of reply parsers.

    Output parsers are objects that help parsing the replies of the underlying
    model into structured data.
    """

    def __init__(self, format_instruction: str = ""):
        """
        Creates an OutputParser.

        :param format_instruction: the instruction for how the response of the
            underlying model should be formatted.
        """
        self._format_instruction = format_instruction

    @property
    def format_instruction(self) -> str:
        """
        Gets the instruction for how the response of the underlying model should
        be formatted.

        :return: the instruction for how the response of the underlying model
            should be formatted, or an empty string if no such instruction.
        """
        return self._format_instruction

    @abstractmethod
    def parse(self,
              reply: str,
              prompt: Optional[Prompt] = None) -> Any:
        """
        Parses the reply of the underlying model to the specified structure.

        :param reply: the reply of the underlying model to the specified
            structure.
        :param prompt: the optional prompt which generates the specified
            reply. The prompt is largely provided in the event the parser
            wants to retry or fix the output in some way, and needs information
            from the prompt to do so.
        :return: the parsing result.
        """
