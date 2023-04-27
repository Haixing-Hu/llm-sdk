# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from llmsdk.common import Example


class LargeLanguageModel(ABC):
    """
    The abstract base class for large language models.
    """
    def __init__(self,
                 max_tokens: int = None,
                 temperature: float = 1.0,
                 top_p: int = 1) -> None:
        """
        Constructs a LargeLanguageModel.

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
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._examples = {}
        self._instruction = ""
        self._logger = logging.getLogger(self.__class__.__name__)

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

    @property
    def instruction(self) -> str:
        """
        The instruction to the model.
        """
        return self._instruction

    @instruction.setter
    def instruction(self, instruction: str) -> None:
        """
        Sets the instruction.
        """
        self._instruction = instruction

    @property
    def examples(self):
        """
        Returns all examples.
        """
        return self._examples

    @examples.setter
    def examples(self, examples: List[Example]) -> None:
        """
        Sets all examples.
        """
        self._examples = examples

    def add_example(self, example: Example) -> None:
        """
        Adds an example to this model.
        """
        self._examples[example.id] = example

    def add_examples(self, examples: List[Example]) -> None:
        """
        Adds an example to this model.
        """
        for example in examples:
            self._examples[example.id] = example

    def remove_example(self, example_id: str) -> None:
        """
        Removes the example with the specific ID.
        """
        if example_id in self._examples:
            del self._examples[example_id]

    def clear_examples(self) -> None:
        """
        Clears all examples of this model.
        """
        self._examples = {}

    def get_example(self, example_id: str) -> Example:
        """
        Gets a single example with the specified ID.
        """
        return self._examples.get(example_id, None)

    def generate(self, prompt: str) -> str:
        """
        Obtains a single generation from this model.

        :param prompt: the prompt.
        :return: the generation.
        """
        generations = self.generate_n(prompt, 1)
        return generations[0]

    def generate_n(self, prompt: str, n: int) -> List[str]:
        """
        Obtains the specified number of generations from this model.

        :param prompt: the prompt.
        :param n: the number of replies to be obtained.
        :return: the list of generations.
        """
        response = self._submit_request(prompt, n)
        return self._parse_response(response)

    @abstractmethod
    def _submit_request(self, prompt: str, n: int) -> Dict[str, Any]:
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
        :return: the list of generations.
        """
