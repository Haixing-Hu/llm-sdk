# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from dataclasses import dataclass

from ..common.data_type import DataType


@dataclass(frozen=True, order=True)
class PayloadSchema:
    """
    The class of schema of a payload field.
    """

    name: str
    """The name of this payload field."""

    type: DataType
    """The type of this payload field."""
