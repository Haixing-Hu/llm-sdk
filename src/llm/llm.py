# ##############################################################################
#                                                                              #
#    Copyright (c) 2023.  Haixing Hu.                                          #
#                                                                              #
#    All rights reserved.                                                      #
#                                                                              #
# ##############################################################################
from abc import ABC, abstractmethod
from .example import Example


class LanguageModel(ABC):
    """
    The abstract base class for language models.
    """
    def __init__(self,
                 model: str = "davinci",
                 temperature: float = 0.7,
                 max_reply_tokens: int = 300) -> None:
        self._examples = {}
        self._instruction = ""
        self._model = model
        self._temperature = temperature
        self._max_reply_tokens = max_reply_tokens

    @property
    def model(self):
        """Returns the model specified for the API."""
        return self._model

    @property
    def temperature(self):
        """Returns the temperature specified for the API."""
        return self._temperature

    @property
    def max_reply_tokens(self):
        """Returns the max tokens of reply specified for the API."""
        return self._max_reply_tokens

    @property
    def instruction(self) -> str:
        """Get the instruction."""
        return self._instruction

    @instruction.setter
    def instruction(self, instruction: str) -> None:
        """Sets the instruction."""
        self._instruction = instruction

    def add_example(self, example: Example) -> None:
        """Adds an example to this engine."""
        self._examples[example.id] = example

    def add_examples(self, examples: list[Example]) -> None:
        """Adds an example to this engine."""
        for example in examples:
            self._examples[example.id] = example

    def remove_example(self, example_id: str) -> None:
        """Remove the example with the specific id."""
        if example_id in self._examples:
            del self._examples[example_id]

    def clear_examples(self) -> None:
        """Clears all examples of this engine."""
        self._examples = {}

    def get_example(self, example_id: str) -> Example:
        """Get a single example."""
        return self._examples.get(example_id, None)

    @property
    def examples(self):
        """Returns all examples."""
        return self._examples

    def get_reply(self, prompt: str) -> str:
        """Obtains the best result as returned by the API."""
        response = self._submit_request(prompt)
        choice = response["choices"][0]
        return self._extract_content(choice)

    @abstractmethod
    def _submit_request(self, prompt: str) -> dict:
        """Calls the OpenAI API with the specified prompt."""
        pass

    @abstractmethod
    def _extract_content(self, choice: dict) -> str:
        """Extracts the content of a choices of a response"""
        pass