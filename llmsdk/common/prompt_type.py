# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from enum import Enum


class PromptType(Enum):
    """
    The enumeration of types of prompts.
    """

    TEXT = "text"
    """
    This type of prompts is used for traditional LLMs. The prompt provided to 
    the LLM is a text, and the LLM will try to complete the input prompt text.
    
    The models use this type of prompt include OpenAI's GPT-1, GPT-2, GPT-3 models.
    """

    MESSAGE = "message"
    """
    This type of prompts is used for conversational LLMs. The prompt provided to 
    the LLM is a list of chatting messages, and the LLM will try to complete 
    the conversation.

    The models use this type of prompt include OpenAI's GPT-3.5, GPT-4, chatGPT 
    models.
    """
