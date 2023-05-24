# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import Optional

from qdrant_client.http import models

from .data_type import DataType
from .distance import Distance
from ..common import Point
from ..criterion import Relation, Operator, Criterion, SimpleCriterion, ComposedCriterion
from ..id_generator import IdGenerator

def to_qdrant_distance(distance: Distance) -> models.Distance:
    """
    Converts the vector distance used in this library into the vector distance
    used in the Qdrant.

    :param distance: the vector distance used in this library.
    :return: the corresponding vector distance used in the Qdrant.
    """
    match distance:
        case Distance.COSINE:
            return models.Distance.COSINE
        case Distance.DOT:
            return models.Distance.DOT
        case Distance.EUCLID:
            return models.Distance.EUCLID
        case _:
            raise ValueError(f"Unsupported distance type: {distance}")


def to_local_distance(distance: models.Distance) -> Distance:
    """
    Converts the vector distance used in the Qdrant into the vector distance
    used in this library.

    :param distance: the vector distance used in the Qdrant.
    :return: the corresponding vector distance used in this library.
    """
    match distance:
        case models.Distance.COSINE:
            return Distance.COSINE
        case models.Distance.DOT:
            return Distance.DOT
        case models.Distance.EUCLID:
            return Distance.EUCLID
        case _:
            raise ValueError(f"Unsupported distance type: {distance}")


def to_qdrant_type(data_type: DataType) -> models.PayloadSchemaType:
    """
    Converts the data type used in this library into the data type used in the
    Qdrant.

    :param data_type: the data type used in this library.
    :return: the corresponding data type used in the Qdrant.
    """
    match data_type:
        case DataType.INT:
            return models.PayloadSchemaType.INTEGER
        case DataType.FLOAT:
            return models.PayloadSchemaType.FLOAT
        case DataType.STRING:
            return models.PayloadSchemaType.KEYWORD
        case _:
            raise ValueError(f"Unsupported data type: {data_type}")


def to_local_type(data_type: models.PayloadSchemaType) -> DataType:
    """
    Converts the data type used in the Qdrant into the data type used in this
    library.

    :param data_type: the data type used in the Qdrant.
    :return: the corresponding data type used in this library.
    """
    match data_type:
        case models.PayloadSchemaType.INTEGER:
            return DataType.INT
        case models.PayloadSchemaType.FLOAT:
            return DataType.FLOAT
        case models.PayloadSchemaType.KEYWORD:
            return DataType.STRING
        case models.PayloadSchemaType.TEXT:
            return DataType.STRING
        case _:
            raise ValueError(f"Unsupported data type: {data_type}")


def to_qdrant_point(point: Point,
                    id_generator: IdGenerator) -> models.PointStruct:
    """
    Converts a Point object into a qdrant PointStruct object.

    :param point: a Point object.
    :param id_generator: the ID generator used to generate ID of documents.
    :return: the converted PointStruct object.
    """
    if point.id is None:
        point.id = id_generator.generate()
    return models.PointStruct(id=point.id,
                              vector=point.vector,
                              payload=point.metadata)


def to_local_point(scored_point: models.ScoredPoint) -> Point:
    """
    Converts a qdrant ScoredPoint object into a Point object.

    :param scored_point: a qdrant ScoredPoint obj  ect.
    :return: the converted Point object.
    """
    return Point(id=scored_point.id,
                 vector=scored_point.vector,
                 metadata=scored_point.payload,
                 score=scored_point.score)


def criterion_to_filter(criterion: Optional[Criterion])\
        -> Optional[models.Filter]:
    if criterion is None:
        return None
    cond = criterion_to_condition(criterion)
    if isinstance(cond, models.Filter):
        return cond
    else:
        return models.Filter(must=[cond])


def criterion_to_condition(criterion: Optional[Criterion]) -> Optional[models.Condition]:
    if criterion is None:
        return None
    if isinstance(criterion, SimpleCriterion):
        return simple_criterion_to_condition(criterion)
    elif isinstance(criterion, ComposedCriterion):
        return composed_criterion_to_filter(criterion)
    else:
        raise ValueError("The criterion must be either a SimpleCriterion or a ComposedCriterion.")


def simple_criterion_to_condition(criterion: Optional[SimpleCriterion]) \
        -> Optional[models.Condition]:
    if criterion is None:
        return None
    match criterion.operator:
        case Operator.EQUAL:
            return models.FieldCondition(key=criterion.property,
                                         match=models.MatchValue(value=criterion.value))
        case Operator.NOT_EQUAL:
            cond = models.FieldCondition(key=criterion.property,
                                         match=models.MatchValue(value=criterion.value))
            return models.Filter(must_not=[cond])
        case Operator.LESS:
            return models.FieldCondition(key=criterion.property,
                                         range=models.Range(lt=criterion.value))
        case Operator.LESS_EQUAL:
            return models.FieldCondition(key=criterion.property,
                                         range=models.Range(lte=criterion.value))
        case Operator.GREATER:
            return models.FieldCondition(key=criterion.property,
                                         range=models.Range(gt=criterion.value))
        case Operator.GREATER_EQUAL:
            return models.FieldCondition(key=criterion.property,
                                         range=models.Range(gte=criterion.value))
        case Operator.IN:
            return models.FieldCondition(key=criterion.property,
                                         match=models.MatchAny(any=criterion.value))
        case Operator.NOT_IN:
            cond = models.FieldCondition(key=criterion.property,
                                         match=models.MatchAny(any=criterion.value))
            return models.Filter(must_not=[cond])
        case Operator.LIKE:
            return models.FieldCondition(key=criterion.property,
                                         match=models.MatchText(text=criterion.value))
        case Operator.NOT_LIKE:
            cond = models.FieldCondition(key=criterion.property,
                                         match=models.MatchText(text=criterion.value))
            return models.Filter(must_not=[cond])
        case Operator.IS_NULL:
            return models.IsNullCondition(is_null=models.PayloadField(key=criterion.property))
        case Operator.NOT_NULL:
            cond = models.IsNullCondition(is_null=models.PayloadField(key=criterion.property))
            return models.Filter(must_not=[cond])
        case _:
            raise ValueError(f"Unsupported comparison operator: {criterion.operator}")


def composed_criterion_to_filter(criterion: Optional[ComposedCriterion]) \
        -> Optional[models.Filter]:
    if criterion is None:
        return None
    match criterion.relation:
        case Relation.AND:
            filters = [criterion_to_condition(c) for c in criterion.criteria]
            return models.Filter(must=filters)
        case Relation.OR:
            filters = [criterion_to_condition(c) for c in criterion.criteria]
            return models.Filter(should=filters)
        case Relation.NOT:
            filters = [criterion_to_condition(c) for c in criterion.criteria]
            return models.Filter(must_not=filters)
        case _:
            raise ValueError(f"Unsupported logic relation: {criterion.relation}")
