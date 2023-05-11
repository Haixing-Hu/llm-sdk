# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from dataclasses import dataclass
from typing import Any, Optional

from .criterion import Criterion
from .operator import Operator


@dataclass(frozen=True)
class SimpleCriterion(Criterion):
    """
    The class represents simple criteria used to filter query result.

    A simple criterion is a comparison expression, which has the form of
    `<prop> <op> <value>`, where <prop> is the path of the property of entities,
    <op> is the comparison operator, and <value> is the value to be compared
    with the property.

    For example::

        user.address.city = "New York"
        user.age > 20

    This class is an immutable class.
    """

    property: str
    """
    The property which matches the filtering condition.
    
    Nested property is supported. For example, "user.address.city"
    """

    operator: Operator
    """
    The comparison operator between the expression.
    """

    value: Optional[Any] = None
    """
    The value used to compare with the property value.
    """
