# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import Any, List

from .relation import Relation
from .operator import Operator
from .simple_criterion import SimpleCriterion
from .composed_criterion import ComposedCriterion


class ComposedCriterionBuilder:
    """
    The builder class used to build query criteria.

    :Example:

    .. code-block:: python

        criterion = ComposedCriterionBuilder(Relation.AND)\
                    .equal("app", expected_app_name)\
                    .like("name", expected_name_substring)\
                    .equal("device.type", expected_device_type)\
                    .less_equal("price", expected_max_price)\
                    .equal("available", True)\
                    .not_null("job.end_time")\
                    .build()
    """

    def __init__(self, relation: Relation) -> None:
        self._logic_relation = relation
        self._criteria = []

    def equal(self, property: str, value: Any):
        criterion = SimpleCriterion(property, Operator.EQUAL, value)
        self._criteria.append(criterion)
        return self

    def not_equal(self, property: str, value: Any):
        criterion = SimpleCriterion(property, Operator.NOT_EQUAL, value)
        self._criteria.append(criterion)
        return self

    def less(self, property: str, value: Any):
        criterion = SimpleCriterion(property, Operator.LESS, value)
        self._criteria.append(criterion)
        return self

    def less_equal(self, property: str, value: Any):
        criterion = SimpleCriterion(property, Operator.LESS_EQUAL, value)
        self._criteria.append(criterion)
        return self

    def greater(self, property: str, value: Any):
        criterion = SimpleCriterion(property, Operator.GREATER, value)
        self._criteria.append(criterion)
        return self

    def greater_equal(self, property: str, value: Any):
        criterion = SimpleCriterion(property, Operator.GREATER_EQUAL, value)
        self._criteria.append(criterion)
        return self

    def is_in(self, property: str, values: List[Any]):
        criterion = SimpleCriterion(property, Operator.IN, values)
        self._criteria.append(criterion)
        return self

    def not_in(self, property: str, values: List[Any]):
        criterion = SimpleCriterion(property, Operator.NOT_IN, values)
        self._criteria.append(criterion)
        return self

    def is_null(self, property: str):
        criterion = SimpleCriterion(property, Operator.IS_NULL)
        self._criteria.append(criterion)
        return self

    def like(self, property: str, value: str):
        criterion = SimpleCriterion(property, Operator.LIKE, value)
        self._criteria.append(criterion)
        return self

    def not_like(self, property: str, value: str):
        criterion = SimpleCriterion(property, Operator.NOT_LIKE, value)
        self._criteria.append(criterion)
        return self

    def not_null(self, property: str):
        criterion = SimpleCriterion(property, Operator.NOT_NULL)
        self._criteria.append(criterion)
        return self

    def require_null(self, property: str, is_null: bool):
        if is_null:
            return self.is_null(property)
        else:
            return self.not_null(property)

    def build(self) -> ComposedCriterion:
        return ComposedCriterion(self._logic_relation, self._criteria)
