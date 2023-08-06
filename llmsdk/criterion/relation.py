# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from enum import Enum


class Relation(Enum):
    """
    The enumeration of logic relations.
    """

    AND = "and"

    OR = "or"

    NOT = "not"
