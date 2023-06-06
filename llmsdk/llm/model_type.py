# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from enum import Enum


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
