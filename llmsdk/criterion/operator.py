# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from enum import Enum


class Operator(Enum):
    """
    The enumeration of comparison operators.
    """

    EQUAL = "="

    NOT_EQUAL = "!="

    LESS = "<"

    LESS_EQUAL = "<="

    GREATER = ">"

    GREATER_EQUAL = ">="

    IN = "in"

    NOT_IN = "not in"

    LIKE = "like"

    NOT_LIKE = "not like"

    IS_NULL = "is null"

    NOT_NULL = "not null"
