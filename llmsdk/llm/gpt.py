# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import Any, Dict, List, Optional

import openai

from .openai import OpenAiModel
from ..common import Example
from llmsdk.llm.openai_utils import (
    check_model_compatibility,
    call_with_retries,
    get_model_tokens,
    count_tokens,
)

DEFAULT_MODEL = "text-davinci-003"


class Gpt(OpenAiModel):
    """
    The class of GPT models from OpenAI.
    """

    def __init__(self,
                 model: str = DEFAULT_MODEL,
                 max_tokens: Optional[int] = None,
                 temperature: float = 1.0,
                 top_p: int = 1,
                 input_prefix: str = "input: ",
                 input_suffix: str = "\n",
                 output_prefix: str = "output: ",
                 output_suffix: str = "\n\n",
                 append_output_prefix: bool = False,
                 api_key: Optional[str] = None,
                 use_proxy: Optional[bool] = None) -> None:
        super().__init__(model=model,
                         max_tokens=max_tokens,
                         temperature=temperature,
                         top_p=top_p,
                         api_key=api_key,
                         use_proxy=use_proxy)
        self._input_prefix = input_prefix
        self._input_suffix = input_suffix
        self._output_prefix = output_prefix
        self._output_suffix = output_suffix
        self._append_output_prefix = append_output_prefix
        self._stop = (output_suffix + input_prefix).strip()
        check_model_compatibility(model=model, endpoint="completions")

    @property
    def input_prefix(self) -> str:
        return self._input_prefix

    @property
    def input_suffix(self) -> str:
        return self._input_suffix

    @property
    def output_prefix(self) -> str:
        return self._output_prefix

    @property
    def output_suffix(self) -> str:
        return self._output_suffix

    @property
    def append_output_prefix(self) -> bool:
        return self._append_output_prefix

    @property
    def stop(self) -> str:
        return self._stop

    def _submit_request(self, prompt: str, n: int) -> Dict[str, Any]:
        full_prompt = self._create_full_prompt(prompt)
        self._logger.debug("Submit a prompt:\n%s", full_prompt)
        if self._max_tokens is None:
            model_tokens = get_model_tokens(model=self._model)
            prompt_tokens = count_tokens(model=self._model, text=full_prompt)
            max_tokens = model_tokens - prompt_tokens
        else:
            max_tokens = self._max_tokens
        self._logger.debug("Max number of generation tokens is: %d", max_tokens)
        response = call_with_retries(openai_api=openai.Completion.create,
                                     model=self._model,
                                     prompt=full_prompt,
                                     max_tokens=max_tokens,
                                     temperature=self._temperature,
                                     top_p=self._top_p,
                                     n=n,
                                     stop=self._stop)
        self._logger.debug("Receive a response:\n%s", response)
        return response

    def _parse_response(self, response: Dict[str, Any]) -> List[str]:
        choices = response["choices"]
        generations = [c["text"] for c in choices]
        return generations

    def _create_full_prompt(self, prompt: str) -> str:
        """
        Creates the criterion for the API request.
        """
        full_prompt = self._instruction + "\n" + self._create_example_prompt() \
                      + self._input_prefix + prompt + self._input_suffix
        if self._append_output_prefix:
            full_prompt = full_prompt + self._output_prefix
        return full_prompt

    def _create_example_prompt(self) -> str:
        """
        Formats all examples to prime the model.
        """
        examples = [self._format_example(e) for e in self.examples.values()]
        return "".join(examples)

    def _format_example(self, example: Example) -> str:
        """
        Formats the input, output pair.
        """
        return self._input_prefix + example.input + self._input_suffix \
            + self._output_prefix + example.output + self._output_suffix
