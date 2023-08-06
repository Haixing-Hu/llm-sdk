# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from enum import Enum
from typing import Dict, Any

from ..prompt import StructuredPromptTemplate
from ..prompt import TextPromptTemplate
from ..prompt import ChatPromptTemplate


class ModelType(Enum):
    """
    The enumeration of types of large language models.
    """

    TEXT_COMPLETION = "text-completion"
    """
    The type of text completion models.
    """

    CHAT_COMPLETION = "chat-completion"
    """
    The type of chat completion models.
    """

    def load_prompt_template(self, config: Dict[str, Any]) -> StructuredPromptTemplate:
        """
        Load the prompt template for this LLM model type.

        :param config: the configuration of the prompt template.
        :return: the prompt template for this LLM model type.
        """
        match self:
            case ModelType.TEXT_COMPLETION:
                template = TextPromptTemplate()
            case ModelType.CHAT_COMPLETION:
                template = ChatPromptTemplate()
            case _:
                raise ValueError(f"Unsupported LLM model type: {self}")
        template.load(config)
        return template
