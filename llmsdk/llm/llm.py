# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .tokenizer import Tokenizer
from ..common import Prompt


class LargeLanguageModel(ABC):
    """
    The abstract base class for large language models.
    """
    def __init__(self,
                 tokenizer: Tokenizer,
                 max_tokens: Optional[int] = None,
                 temperature: float = 1.0,
                 top_p: int = 1) -> None:
        """
        Constructs a LargeLanguageModel.

        :param tokenizer: the tokenizer used by this LLM.
        :param max_tokens: the maximum number of tokens of the generated message.
            If it is None, the value will be calculated automatically.
        :param temperature: What sampling temperature to use. Higher values like
            0.8 will make the output more random, while lower values like 0.2
            will make it more focused and deterministic. We generally recommend
            altering this or top_p but not both.
        :param top_p: An alternative to sampling with temperature, called
            nucleus sampling, where the model considers the results of the
            tokens with top_p probability mass. So 0.1 means only the tokens
            comprising the top 10% probability mass are considered. We generally
            recommend altering this or temperature but not both.
        """
        self._tokenizer = tokenizer
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    @property
    def max_tokens(self):
        """
        The maximum number of tokens to generate in the completion.

        The token count of your prompt plus max_tokens cannot exceed the model's
        context length.
        """
        return self._max_tokens

    @property
    def temperature(self):
        """
        What sampling temperature to use, between 0 and 2. Higher values like
        0.8 will make the output more random, while lower values like 0.2 will
        make it more focused and deterministic.

        We generally recommend altering this or top_p but not both.
        """
        return self._temperature

    @property
    def top_p(self) -> int:
        """
        An alternative to sampling with temperature, called nucleus sampling,
        where the model considers the results of the tokens with top_p
        probability mass. So 0.1 means only the tokens comprising the top 10%
        probability mass are considered.

        We generally recommend altering this or temperature but not both.
        """
        return self._top_p

    def generate(self, prompt: Prompt) -> str:
        """
        Generates a single reply from this model.

        :param prompt: the prompt.
        :return: the generated reply.
        """
        generations = self.generate_n(prompt, 1)
        return generations[0]

    def generate_n(self, prompt: Prompt, n: int) -> List[str]:
        """
        Generates the specified number of top replies from this model.

        :param prompt: the prompt.
        :param n: the number of replies to be obtained.
        :return: the list of replies.
        """
        response = self._submit_request(prompt, n)
        return self._parse_response(response)

    @abstractmethod
    def _submit_request(self, prompt: Prompt, n: int) -> Dict[str, Any]:
        """
        Calls the underlying model with the specified prompt.

        :param prompt: the prompt.
        :param n: the number of replies to be obtained.
        :return: the response of the model.
        """

    @abstractmethod
    def _parse_response(self, response: Dict[str, Any]) -> List[str]:
        """
        Parses the replies from the response of the model.

        :param response: the response.
        :return: the list of replies.
        """
