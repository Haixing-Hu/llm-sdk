# ##############################################################################
#                                                                              #
#    Copyright (c) 2023.  Haixing Hu.                                          #
#                                                                              #
#    All rights reserved.                                                      #
#                                                                              #
# ##############################################################################
import openai
import tiktoken
from .openai_model import OpenAiModel

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
    def __int__(self,
                model: str = DEFAULT_MODEL,
                max_tokens: int = None,
                temperature: float = 1.0,
                top_p: int = 1) -> None:
        super().__int__(model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p)
        if model not in COMPATIBLE_MODELS:
            raise ValueError(f"Incompatible model {model}. "
                             f"The compatible models are: {COMPATIBLE_MODELS}")

    def _submit_request(self, prompt: str, n: int) -> dict:
        messages = self._create_messages(prompt)
        self._logger.debug("Submit messages:\n{}", messages)
        if self._max_tokens is None:
            messages_tokens = self.count_tokens_of_messages(messages)
            model_tokens = OpenAiModel.get_model_tokens(self._model)
            max_tokens = model_tokens - messages_tokens
        else:
            max_tokens = self._max_tokens
        self._logger.debug("Max number of generation tokens is: {}", max_tokens)
        # FIXME: deal with errors and retries
        response = openai.ChatCompletion.create(engine=self.model,
                                                messages=messages,
                                                max_tokens=max_tokens,
                                                temperature=self._temperature,
                                                top_p=self._top_p,
                                                n=n)
        self._logger.debug("Receive a response:\n{}", response)
        return response

    def _parse_response(self, response: dict) -> list[str]:
        choices = response["choices"]
        replies = [c["content"] for c in choices]
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

    def count_tokens_of_messages(self, messages: list[dict], model):
        """
        Counts the number of tokens used by a list of messages.

        Reference: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        """
        if model == "gpt-3.5-turbo":
            self._logger.warn("gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
            return self.count_tokens_of_messages(messages, model="gpt-3.5-turbo-0301")
        elif model == "gpt-4":
            self._logger.warn("gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
            return self.count_tokens_of_messages(messages, model="gpt-4-0314")
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1    # if there's a name, the role is omitted
        elif model == "gpt-4-0314":
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(f"""
                ChatGpt._count_tokens_of_messages() is not implemented for model {model}. 
                See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are 
                converted to tokens.
            """)
        codec = tiktoken.encoding_for_model(model)
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(codec.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens
