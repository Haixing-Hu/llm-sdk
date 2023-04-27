# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from abc import ABC

from llmsdk.llm import LargeLanguageModel
from llmsdk.util.openai_utils import init_openai


class OpenAiModel(LargeLanguageModel, ABC):
    """
    The base class of models from OpenAI.
    """

    def __init__(self,
                 model: str,
                 max_tokens: int,
                 temperature: float,
                 top_p: int,
                 api_key: str,
                 use_proxy: bool) -> None:
        """
        Create a OpenAiModel.

        :param model: the name of the OpenAI model.
        :param max_tokens: the maximum number of tokens in the reply of the
            OpenAI's model. If it is None, the value will be calculated
            automatically.
        :param temperature: What sampling temperature to use, between 0 and 2.
            Higher values like 0.8 will make the output more random, while lower
            values like 0.2 will make it more focused and deterministic. We
            generally recommend altering this or top_p but not both.
        :param top_p: An alternative to sampling with temperature, called
            nucleus sampling, where the model considers the results of the
            tokens with top_p probability mass. So 0.1 means only the tokens
            comprising the top 10% probability mass are considered. We generally
            recommend altering this or temperature but not both.
        :param api_key: the OpenAI API key.
        :param use_proxy: whether to use the proxy.
        """
        super().__init__(max_tokens=max_tokens,
                         temperature=temperature,
                         top_p=top_p)
        self._model = model
        init_openai(api_key=api_key,
                    use_proxy=use_proxy)

    @property
    def model(self) -> str:
        """
        The name of model of this OpenAI LLM.
        """
        return self._model
