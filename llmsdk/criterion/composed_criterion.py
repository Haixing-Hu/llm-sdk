# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from dataclasses import dataclass
from typing import List

from .criterion import Criterion
from .relation import Relation
from ..common.metadata import Metadata


@dataclass(frozen=True)
class ComposedCriterion(Criterion):
    """
    The class of criteria which combines sub-criteria with logic relations.

    For example::

        `user.address.city = "New York" AND user.age > 20`

    This class is an immutable class.
    """

    relation: Relation
    """The logic relation between sub-criteria."""

    criteria: List[Criterion]
    """The list of sub-criteria"""

    def test(self, metadata: Metadata) -> bool:
        match self.relation:
            case Relation.AND:
                for criterion in self.criteria:
                    if not criterion.test(metadata):
                        return False
                return True
            case Relation.OR:
                for criterion in self.criteria:
                    if criterion.test(metadata):
                        return True
                return False
            case Relation.NOT:
                if len(self.criteria) != 1:
                    raise ValueError("The number of criteria for NOT relation must be 1.")
                return not self.criteria[0].test(metadata)
            case _:
                raise ValueError(f"Unsupported relation: {self.relation}")
