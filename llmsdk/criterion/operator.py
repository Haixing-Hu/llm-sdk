# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from enum import Enum
from typing import Any, Type
import re


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

    def test(self, lhs: Any, rhs: Any = None) -> bool:
        """
        Tests whether two values fulfill this operator.

        :param lhs: the left hand side of the operator.
        :param rhs: the right hand side of the operator. If this operator is a
            unary operator (i.e., `IS_NULL` and `NOT_NULL`), this argument should
            be `None`.
        :return: `True` if `lhs` and `rhs` fulfill this operator; `False` otherwise.
        """
        match self:
            case Operator.EQUAL:
                return lhs == rhs
            case Operator.NOT_EQUAL:
                return lhs != rhs
            case Operator.LESS:
                return lhs < rhs
            case Operator.LESS_EQUAL:
                return lhs <= rhs
            case Operator.GREATER:
                return lhs > rhs
            case Operator.GREATER_EQUAL:
                return lhs >= rhs
            case Operator.IN:
                Operator._ensure_type("rhs", rhs, list)
                return lhs in rhs
            case Operator.NOT_IN:
                Operator._ensure_type("rhs", rhs, list)
                return lhs not in rhs
            case Operator.LIKE:
                Operator._ensure_type("lhs", lhs, str)
                Operator._ensure_type("rhs", rhs, str)
                pattern = re.escape(rhs).replace("%", ".*")
                return re.search(pattern, lhs) is not None
            case Operator.NOT_LIKE:
                Operator._ensure_type("lhs", lhs, str)
                Operator._ensure_type("rhs", rhs, str)
                pattern = re.escape(rhs).replace("%", ".*")
                return re.search(pattern, lhs) is None
            case Operator.IS_NULL:
                return lhs is None
            case Operator.NOT_NULL:
                return lhs is not None
            case _:
                raise ValueError(f"Unsupported operator: {self}")

    @classmethod
    def _ensure_type(cls, name: str, value: Any, data_type: Type):
        if type(value) != data_type:
            raise ValueError(f"The argument {name} must be a {data_type}: {value}")
