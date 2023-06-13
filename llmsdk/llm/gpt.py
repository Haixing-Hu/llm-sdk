# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import Any, Dict, List, Optional

from ..common import Prompt
from ..util.openai_utils import (
    check_model_compatibility,
    call_with_retries,
    get_model_tokens,
)
from .openai import OpenAiModel
from .model_type import ModelType

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
                 api_key: Optional[str] = None,
                 use_proxy: Optional[bool] = None) -> None:
        super().__init__(model=model,
                         model_type=ModelType.TEXT_COMPLETION,
                         max_tokens=max_tokens,
                         temperature=temperature,
                         top_p=top_p,
                         api_key=api_key,
                         use_proxy=use_proxy)
        check_model_compatibility(model=model, endpoint="completions")
        try:
            import openai
        except ImportError:
            raise ImportError("Openai Python package is not installed, please "
                              "install it with `pip install openai`.")
        self._api = openai.Completion.create

    def _submit_request(self, prompt: Prompt, n: int) -> Dict[str, Any]:
        if not isinstance(prompt, str):
            raise ValueError("The OpenAI's GPT model only support text prompt.")
        self._logger.debug("Submit a prompt:\n%s", prompt)
        if self._max_tokens is None:
            model_tokens = get_model_tokens(model=self._model)
            prompt_tokens = self._tokenizer.count_text_tokens(prompt)
            max_tokens = model_tokens - prompt_tokens
        else:
            max_tokens = self._max_tokens
        self._logger.debug("Max number of generation tokens is: %d", max_tokens)
        response = call_with_retries(openai_api=self._api,
                                     model=self._model,
                                     prompt=prompt,
                                     max_tokens=max_tokens,
                                     temperature=self._temperature,
                                     top_p=self._top_p,
                                     n=n)
        self._logger.debug("Receive a response:\n%s", response)
        return response

    def _parse_response(self, response: Dict[str, Any]) -> List[str]:
        choices = response["choices"]
        generations = [c["text"] for c in choices]
        return generations
