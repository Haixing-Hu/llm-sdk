# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from dataclasses import dataclass


@dataclass
class ChatMessage:
    """
    The data structure represents chatting messages.
    """

    role: str
    """
    The role of the speaker.
    """

    content: str
    """
    The content of the message.
    """
