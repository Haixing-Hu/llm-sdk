# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from enum import Enum
from frozendict import frozendict


class Role(Enum):
    """
    The enumeration of roles of a human-AI conversation.
    """
    HUMAN = "Human"

    AI = "AI"

    SYSTEM = "System"


ROLE_NAMES_MAP = frozendict({e: e.value for e in Role})
"""
A readonly map mapping a enumerator of the enumeration class Role into its name.
"""
