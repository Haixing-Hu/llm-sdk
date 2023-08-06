# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from enum import Enum


class DataType(Enum):
    """
    The enumeration of supported data types of a field.
    """

    INT = int
    """The data type of 64-bits integer numbers."""

    FLOAT = float
    """The data type of 64-bits floating point numbers."""

    STRING = str
    """The data type of string values."""
