# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from dataclasses import dataclass
from typing import List

from .criterion import Criterion
from .relation import Relation


@dataclass(frozen=True)
class ComposedCriterion(Criterion):
    """
    The class of criteria which combines sub-criteria with logic relations.

    For example::

        user.address.city = "New York" AND user.age > 20

    This class is an immutable class.
    """

    relation: Relation
    """The logic relation between sub-criteria."""

    criteria: List[Criterion]
    """The list of sub-criteria"""
