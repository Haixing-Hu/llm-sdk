# ##############################################################################
#                                                                              #
#    Copyright (c) 2023.  Haixing Hu.                                          #
#                                                                              #
#    All rights reserved.                                                      #
#                                                                              #
# ##############################################################################
import openai
from .openai_llm import OpenAiModel
from ..util.openai_utils import (
    check_model_compatibility,
    call_with_retries,
    count_message_tokens,
    get_model_tokens,
)

COMPATIBLE_MODELS = [
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-32k",
    "gpt-4-32k-0314",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
]

DEFAULT_MODEL = "gpt-3.5-turbo"


class ChatGpt(OpenAiModel):
    """
    The class of chatGPT models from OpenAI.
    """
    def __init__(self,
                 model: str = DEFAULT_MODEL,
                 max_tokens: int = None,
                 temperature: float = 1.0,
                 top_p: int = 1,
                 api_key: str = None,
                 use_proxy: bool = None) -> None:
        super().__init__(model=model,
                         max_tokens=max_tokens,
                         temperature=temperature,
                         top_p=top_p,
                         api_key=api_key,
                         use_proxy=use_proxy)
        check_model_compatibility(model=model, endpoint="chat-completions")

    def _submit_request(self, prompt: str, n: int) -> dict:
        messages = self._create_messages(prompt)
        self._logger.debug("Submit messages:\n%s", messages)
        if self._max_tokens is None:
            model_tokens = get_model_tokens(model=self._model)
            messages_tokens = count_message_tokens(model=self._model,
                                                   messages=messages)
            max_tokens = model_tokens - messages_tokens
        else:
            max_tokens = self._max_tokens
        self._logger.debug("Max number of generation tokens is: %d", max_tokens)
        response = call_with_retries(openai_api=openai.ChatCompletion.create,
                                     model=self._model,
                                     messages=messages,
                                     max_tokens=max_tokens,
                                     temperature=self._temperature,
                                     top_p=self._top_p,
                                     n=n)
        self._logger.debug("Receive a response:\n%s", response)
        return response

    def _parse_response(self, response: dict) -> list[str]:
        choices = response["choices"]
        replies = [c["message"]["content"] for c in choices]
        return replies

    def _create_messages(self, prompt: str) -> list[dict]:
        """
        Creates the list of messages for the API request.
        """
        messages = []
        if len(self._instruction) > 0:
            messages.append({"role": "system", "content": self._instruction})
        for example in self._examples.values():
            messages.append({"role": "user", "content": example.input})
            messages.append({"role": "assistant", "content": example.output})
        messages.append({"role": "user", "content": prompt})
        return messages