# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from enum import Enum


class Relation(Enum):
    """
    The enumeration of logic relations.
    """

    AND = "and"

    OR = "or"

    NOT = "not"
