# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from typing import List
import logging

from ...common.role import Role
from ...common.message import Message
from .tokernizer import Tokenizer, SpecialTokenSet


OPENAI_ROLE_NAMES_MAP = {
    Role.SYSTEM: "system",
    Role.HUMAN: "user",
    Role.AI: "assistant"
}

IMPORT_TIKTOKEN_ERROR_MESSAGE = "Tiktoken is not installed, please install it " \
                                "with `pip install tiktoken`."


class OpenAiTokenizer(Tokenizer):
    """
    The Tokenizer implemented with OpenAI's tiktoken.
    """

    def __init__(self, model: str) -> None:
        """
        Creates a TiktokenTokenizer.

        :param model: the model name of the OpenAI's LLM.
        """
        self._model = model
        self._logger = logging.getLogger(self.__class__.__name__)
        try:
            import tiktoken
        except ImportError:
            raise ImportError(IMPORT_TIKTOKEN_ERROR_MESSAGE)
        self._codec = tiktoken.encoding_for_model(self._model)

    def encode(self,
               text: str,
               allowed_special: SpecialTokenSet = None,
               disallowed_special: SpecialTokenSet = "all") -> List[int]:
        return self._codec.encode(text,
                                  allowed_special=allowed_special or set(),
                                  disallowed_special=disallowed_special or set())

    def decode(self,
               tokens: List[int],
               errors: str = "replace") -> str:
        return self._codec.decode(tokens, errors)

    def count_message_tokens(self,
                             messages: List[Message],
                             allowed_special: SpecialTokenSet = None,
                             disallowed_special: SpecialTokenSet = "all") -> int:
        return self._count_message_tokens_impl(model=self._model,
                                               messages=messages,
                                               allowed_special=allowed_special or set(),
                                               disallowed_special=disallowed_special or set())

    def _count_message_tokens_impl(self,
                                   model: str,
                                   messages: List[Message],
                                   allowed_special: SpecialTokenSet,
                                   disallowed_special: SpecialTokenSet) -> int:
        """
        Counts the number of tokens encoded from the list of chatting messages.

        Reference: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

        :param model: the model name of the OpenAI's LLM.
        :param messages: the list of chatting messages.
        :param allowed_special: the set of special tokens allowed in the text.
        :param disallowed_special: the set of special tokens disallowed in the
            text.
        :return: the number of tokens encoded from the list of chatting messages.
        """
        if model == "gpt-3.5-turbo":
            self._logger.warning("gpt-3.5-turbo may change over time. "
                                 "Returning num tokens assuming gpt-3.5-turbo-0301.")
            return self._count_message_tokens_impl(model="gpt-3.5-turbo-0301",
                                                   messages=messages,
                                                   allowed_special=allowed_special,
                                                   disallowed_special=disallowed_special)
        elif model == "gpt-4":
            self._logger.warning("gpt-4 may change over time. "
                                 "Returning num tokens assuming gpt-4-0314.")
            return self._count_message_tokens_impl(model="gpt-4-0314",
                                                   messages=messages,
                                                   allowed_special=allowed_special,
                                                   disallowed_special=disallowed_special)
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif model == "gpt-4-0314":
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(f"""
                    ChatGpt._count_tokens_of_messages() is not implemented for 
                    model {model}. 
                    See https://github.com/openai/openai-python/blob/main/chatml.md 
                    for information on how messages are converted to tokens.
                """)

        import tiktoken
        codec = tiktoken.encoding_for_model(model)
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            msg_dict = message.to_dict(role_names_map=OPENAI_ROLE_NAMES_MAP)
            for key, value in msg_dict.items():
                tokens = codec.encode(value,
                                      allowed_special=allowed_special,
                                      disallowed_special=disallowed_special)
                num_tokens += len(tokens)
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens
