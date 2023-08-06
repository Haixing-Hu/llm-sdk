# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from abc import ABC, abstractmethod
from typing import List, Union, AbstractSet, Literal

from ...common.message import MessageList


SpecialTokenSet = Union[Literal["all"], AbstractSet[str], type(None)]
"""
The type for special token set.
"""


class Tokenizer(ABC):
    """
    The interface of tokenizers for LLMs.
    """

    @abstractmethod
    def encode(self,
               text: str,
               allowed_special: SpecialTokenSet = None,
               disallowed_special: SpecialTokenSet = "all") -> List[int]:
        """
        Encodes a piece of text to list of tokens.

        Special tokens are artificial tokens used to unlock capabilities from a
        model, such as fill-in-the-middle. So we want to be careful about
        accidentally encoding special tokens, since they can be used to trick a
        model into doing something we don't want it to do.

        Hence, by default, encode will raise an error if it encounters text that
        corresponds to a special token. This can be controlled on a per-token
        level using the `allowed_special` and `disallowed_special` parameters.

        In particular:

        - Setting `disallowed_special` to () will prevent this function from
          raising errors and cause all text corresponding to special tokens to
          be encoded as natural text.
        - Setting `allowed_special` to "all" will cause this function to treat
          all text corresponding to special tokens to be encoded as special
          tokens.

        :param text: the specified text to be encoded.
        :param allowed_special: the set of special tokens allowed in the text.
        :param disallowed_special: the set of special tokens disallowed in the
            text.
        :return: the list of tokens encoded from the specified text.
        """

    @abstractmethod
    def decode(self,
               tokens: List[int],
               errors: str = "replace") -> str:
        """
        Decodes a list of tokens into the text.

        :param tokens: the specified list of tokens to be decoded.
        :param errors: controls how Unicode decoding errors are handled. If
            'strict' (the default), a UnicodeError exception is raised. Other
            possible values are 'ignore', 'replace', and any other name
            registered via codecs.register_error(). See `bytes.decode()`.
        :return: the text decoded from the specified list of tokens.
        """

    def count_text_tokens(self,
                          text: str,
                          allowed_special: SpecialTokenSet = None,
                          disallowed_special: SpecialTokenSet = "all") -> int:
        """
        Counts the number of tokens encoded from the specified piece of text.

        :param text: the specified piece of text.
        :param allowed_special: the set of special tokens allowed in the text.
        :param disallowed_special: the set of special tokens disallowed in the
            text.
        :return: the number of tokens encoded from the specified text.
        """
        tokens = self.encode(text,
                             allowed_special=allowed_special,
                             disallowed_special=disallowed_special)
        return len(tokens)

    @abstractmethod
    def count_message_tokens(self,
                             messages: MessageList,
                             allowed_special: SpecialTokenSet = None,
                             disallowed_special: SpecialTokenSet = "all") -> int:
        """
        Counts the number of tokens encoded from the list of chatting messages.

        :param messages: the list of chatting messages.
        :param allowed_special: the set of special tokens allowed in the text.
        :param disallowed_special: the set of special tokens disallowed in the
            text.
        :return: the number of tokens encoded from the list of chatting messages.
        """
