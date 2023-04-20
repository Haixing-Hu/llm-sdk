# ##############################################################################
#                                                                              #
#    Copyright (c) 2023.  Haixing Hu.                                          #
#                                                                              #
#    All rights reserved.                                                      #
#                                                                              #
# ##############################################################################
# ##############################################################################
#                                                                              #
#    Copyright (c) 2023.  Haixing Hu.                                          #
#                                                                              #
#    All rights reserved.                                                      #
#                                                                              #
# ##############################################################################
from abc import ABC
import openai
import tiktoken
from .llm import LargeLanguageModel

MODEL_TOKEN_MAPPING = {
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0314": 32768,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0301": 4096,
    "text-ada-001": 2049,
    "ada": 2049,
    "text-babbage-001": 2040,
    "babbage": 2049,
    "text-curie-001": 2049,
    "curie": 2049,
    "davinci": 2049,
    "text-davinci-003": 4097,
    "text-davinci-002": 4097,
    "code-davinci-002": 8001,
    "code-davinci-001": 8001,
    "code-cushman-002": 2048,
    "code-cushman-001": 2048,
}


class OpenAiModel(LargeLanguageModel, ABC):
    """
    The base class of models from OpenAI.
    """
    def __int__(self,
                model: str,
                max_tokens: int,
                temperature: float,
                top_p: int) -> None:
        super().__int__(max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p)
        self._model = model

    @classmethod
    def set_api_key(cls, key) -> None:
        """
        Sets the OpenAI API key.
        """
        openai.api_key = key

    @classmethod
    def get_model_tokens(cls, model) -> int:
        """
        Gets the context length in tokens of the specified model.

        :param model: the name of the model.
        :return: the context length in tokens of the specified model.
        """
        return MODEL_TOKEN_MAPPING[model]

    @property
    def model(self) -> str:
        """
        The name of model of this OpenAI LLM.
        """
        return self._model

    def count_tokens(self, text) -> int:
        codec = tiktoken.encoding_for_model(self._model)
        tokenized_text = codec.encode(text)
        return len(tokenized_text)
