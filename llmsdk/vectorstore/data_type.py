# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from enum import Enum


class DataType(Enum):
    """
    The enumeration of supported data types of a field.
    """

    INT = int
    """The data type of 64-bits integer numbers."""

    FLOAT = float
    """The data type of 64-bits floating point numbers."""

    BOOL = bool
    """The data type of boolean values."""

    STRING = str
    """The data type of string values."""
