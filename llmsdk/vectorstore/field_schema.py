# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from dataclasses import dataclass, field

from .data_type import DataType


@dataclass(frozen=True)
class FieldSchema:
    """
    The class of schema of a field.
    """

    name: str
    """The name of the field."""

    type: DataType
    """The type of the field."""

    indexed: bool
    """Indicates whether this field should be indexed."""
