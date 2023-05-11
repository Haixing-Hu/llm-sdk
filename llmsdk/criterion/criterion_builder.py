# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import Any, List, Dict

from .operator import Operator
from .relation import Relation
from .simple_criterion import SimpleCriterion
from .composed_criterion import ComposedCriterion


def equal(property: str, value: Any) -> SimpleCriterion:
    """
    Builds a simple query criterion where the value of the specified property
    is equal to the specified value.

    :param property: the path of the specified property.
    :param value: the specified value.
    :return: a simple query criterion where the value of the specified property
    is equal to the specified value.
    """
    return SimpleCriterion(property, Operator.EQUAL, value)


def not_equal(property: str, value: Any) -> SimpleCriterion:
    """
    Builds a simple query criterion where the value of the specified property is
    not equal to the specified value.

    :param property: the path of the specified property.
    :param value: the specified value.
    :return: a simple query criterion where the value of the specified property
    is not equal to the specified value.
    """
    return SimpleCriterion(property, Operator.NOT_EQUAL, value)


def less(property: str, value: Any) -> SimpleCriterion:
    """
    Builds a simple query criterion where the value of the specified property is
    less than the specified value.

    :param property: the path of the specified property.
    :param value: the specified value.
    :return: a simple query criterion where the value of the specified property
    is less than the specified value.
    """
    return SimpleCriterion(property, Operator.LESS, value)


def less_equal(property: str, value: Any) -> SimpleCriterion:
    """
    Builds a simple query criterion where the value of the specified property
    is less than or equal to the specified value.

    :param property: the path of the specified property.
    :param value: the specified value.
    :return: a simple query criterion where the value of the specified property
    is less than or equal to the specified value.
    """
    return SimpleCriterion(property, Operator.LESS_EQUAL, value)


def greater(property: str, value: Any) -> SimpleCriterion:
    """
    Builds a simple query criterion where the value of the specified property is
    greater than the specified value.

    :param property: the path of the specified property.
    :param value: the specified value.
    :return: a simple query criterion where the value of the specified property
    is greater than the specified value.
    """
    return SimpleCriterion(property, Operator.GREATER, value)


def greater_equal(property: str, value: Any) -> SimpleCriterion:
    """
    Builds a simple query criterion where the value of the specified property is
    greater than or equal to the specified value.

    :param property: the path of the specified property.
    :param value: the specified value.
    :return: a simple query criterion where the value of the specified property
    is greater than or equal to the specified value.
    """
    return SimpleCriterion(property, Operator.GREATER_EQUAL, value)


def is_in(property: str, values: List[Any]) -> SimpleCriterion:
    """
    Builds a simple query criterion where the value of the specified property is
    contained in the specified list of values.

    :param property: the path of the specified property.
    :param values: the specified list of values.
    :return: a simple query criterion where the value of the specified property
    is contained in the list of the specified values.
    """
    return SimpleCriterion(property, Operator.IN, values)


def not_in(property: str, values: List[Any]) -> SimpleCriterion:
    """
    Builds a simple query criterion where the value of the specified property is
    not contained in the specified list of values.

    :param property: the path of the specified property.
    :param values: the specified list of values.
    :return: a simple query criterion where the value of the specified property
    is not contained in the list of the specified values.
    """
    return SimpleCriterion(property, Operator.NOT_IN, values)


def like(property: str, value: str) -> SimpleCriterion:
    """
    Builds a simple query criterion where the value of the specified property
    contains the specified value as a substring.

    :param property: the path of the specified property.
    :param value: the specified value.
    :return: a simple query criterion where the value of the specified property
    contains the specified value as a substring.
    """
    return SimpleCriterion(property, Operator.LIKE, value)


def not_like(property: str, value: str) -> SimpleCriterion:
    """
    Builds a simple query criterion where the value of the specified property
    does not contain the specified value as a substring.

    :param property: the path of the specified property.
    :param value: the specified value.
    :return: a simple query criterion where the value of the specified property
    does not contain the specified value as a substring.
    """
    return SimpleCriterion(property, Operator.NOT_LIKE, value)


def is_null(property: str) -> SimpleCriterion:
    """
    Builds a simple query criterion where the value of the specified property is
    null.

    :param property: the path of the specified property.
    :return: a simple query criterion where the value of the specified property
    is null.
    """
    return SimpleCriterion(property, Operator.IS_NULL)


def not_null(property: str) -> SimpleCriterion:
    """
    Builds a simple query criterion where the value of the specified property is
    not null.

    :param property: the path of the specified property.
    :return: a simple query criterion where the value of the specified property
    is not null.
    """
    return SimpleCriterion(property, Operator.NOT_NULL)


def all_equal(params: Dict[str, Any]) -> ComposedCriterion:
    """
    Builds a composed query criterion where all values of the specified properties
    is equal to the corresponding values. The properties and expected values are
    specified with a dict argument.

    :param params: the dict specifies the property/value pairs.
    :return: a composed query criterion where all values of the specified properties
    is equal to the corresponding values.
    """
    criteria = [equal(property, value) for property, value in params.items()]
    return ComposedCriterion(Relation.AND, criteria)


def any_equal(params: Dict[str, Any]) -> ComposedCriterion:
    """
    Builds a composed query criterion where at least one value of the specified
    properties is equal to the corresponding values. The properties and expected
    values are specified with a dict argument.

    :param params: the dict specifies the property/value pairs.
    :return: a composed query criterion where at least one value of the specified
    properties is equal to the corresponding values.
    """
    criteria = [equal(property, value) for property, value in params.items()]
    return ComposedCriterion(Relation.OR, criteria)
