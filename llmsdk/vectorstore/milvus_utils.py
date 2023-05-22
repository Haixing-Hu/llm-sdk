# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import Optional

import pymilvus

from .data_type import DataType
from .distance import Distance
from .payload_schema import PayloadSchema
from ..criterion import Criterion, SimpleCriterion, ComposedCriterion, Operator, Relation


def to_milvus_distance(distance: Distance) -> str:
    """
    Converts the enumeration of distance used in our library into the enumeration
    of distance used in the milvus library.

    :param distance: the enumeration of distance used in our library.
    :return: the corresponding enumeration of distance used in the milvus library.
    """
    match distance:
        case Distance.COSINE:
            return "IP"
        case Distance.DOT:
            return "IP"
        case Distance.EUCLID:
            return "L2"
        case _:
            raise ValueError(f"Unsupported distance type: {distance}")


def to_local_distance(distance: str) -> Distance:
    """
    Converts the enumeration of distance used in the milvus library into the
    enumeration of distance used in this library.

    :param distance: the enumeration of distance used in the milvus library.
    :return: the corresponding enumeration of distance used in this library.
    """
    match distance:
        case "IP":
            return Distance.COSINE
        case "L2":
            return Distance.EUCLID
        case _:
            raise ValueError(f"Unsupported distance type: {distance}")


def to_milvus_type(data_type: DataType) -> pymilvus.DataType:
    match data_type:
        case DataType.INT:
            return pymilvus.DataType.INT64
        case DataType.FLOAT:
            return pymilvus.DataType.FLOAT
        case DataType.STRING:
            return pymilvus.DataType.STRING
        case _:
            raise ValueError(f"Unsupported data type: {data_type}")


def to_local_type(data_type: pymilvus.DataType) -> DataType:
    match data_type:
        case pymilvus.DataType.INT8:
            return DataType.INT
        case pymilvus.DataType.INT16:
            return DataType.INT
        case pymilvus.DataType.INT32:
            return DataType.INT
        case pymilvus.DataType.INT64:
            return DataType.INT
        case pymilvus.DataType.FLOAT:
            return DataType.FLOAT
        case pymilvus.DataType.DOUBLE:
            return DataType.FLOAT
        case pymilvus.DataType.STRING:
            return DataType.STRING
        case pymilvus.DataType.VARCHAR:
            return DataType.STRING
        case _:
            raise ValueError(f"Unsupported data type: {data_type}")


def to_milvus_field_schema(schema: PayloadSchema) -> pymilvus.FieldSchema:
    return pymilvus.FieldSchema(name=schema.name,
                                dtype=to_milvus_type(schema.type))


def criterion_to_expr(criterion: Optional[Criterion]) -> Optional[str]:
    if criterion is None:
        return None
    if isinstance(criterion, SimpleCriterion):
        return simple_criterion_to_expr(criterion)
    elif isinstance(criterion, ComposedCriterion):
        return composed_criterion_to_expr(criterion)
    else:
        raise ValueError("The criterion must be either a SimpleCriterion or a ComposedCriterion.")


def simple_criterion_to_expr(criterion: Optional[SimpleCriterion]) -> Optional[str]:
    if criterion is None:
        return None
    match criterion.operator:
        case Operator.EQUAL:
            return f"{criterion.property} == {criterion.value}"
        case Operator.NOT_EQUAL:
            return f"{criterion.property} != {criterion.value}"
        case Operator.LESS:
            return f"{criterion.property} < {criterion.value}"
        case Operator.LESS_EQUAL:
            return f"{criterion.property} <= {criterion.value}"
        case Operator.GREATER:
            return f"{criterion.property} > {criterion.value}"
        case Operator.GREATER_EQUAL:
            return f"{criterion.property} >= {criterion.value}"
        case Operator.IN:
            return f"{criterion.property} in {criterion.value}"
        case Operator.NOT_IN:
            return f"{criterion.property} not in {criterion.value}"
        case Operator.LIKE:
            return f"{criterion.property} like \"{criterion.value}\""
        case Operator.NOT_LIKE:
            return f"{criterion.property} not like \"{criterion.value}\""
        # FIXME: IS_NULL and NOT_NULL is not supported
        case _:
            raise ValueError(f"Unsupported comparison operator: {criterion.operator}")


def composed_criterion_to_expr(criterion: Optional[ComposedCriterion]) -> Optional[str]:
    if criterion is None:
        return None
    match criterion.relation:
        case Relation.AND:
            n = len(criterion.criteria)
            if n == 0:
                return None
            elif n == 1:
                return criterion_to_expr(criterion.criteria[0])
            else:
                exprs = [f"({criterion_to_expr(c)})" for c in criterion.criteria]
                return " and ".join(exprs)
        case Relation.OR:
            n = len(criterion.criteria)
            if n == 0:
                return None
            elif n == 1:
                return criterion_to_expr(criterion.criteria[0])
            else:
                exprs = [f"({criterion_to_expr(c)})" for c in criterion.criteria]
                return " or ".join(exprs)
        case Relation.NOT:
            n = len(criterion.criteria)
            if n == 0:
                return None
            elif n == 1:
                return f"not ({criterion_to_expr(criterion.criteria[0])})"
            else:
                raise ValueError("ComposedCriterion with NOT relation should "
                                 "have one sub-criterion.")
        case _:
            raise ValueError(f"Unsupported logic relation: {criterion.relation}")
