# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import Optional, Any, List
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models

from .vector_store import VectorStore
from ..common import Vector, Point
from ..criterion import Relation, Operator, Criterion, SimpleCriterion, ComposedCriterion


class QdrantVectorStore(VectorStore):
    """
    The vector store based on the Qdrant vector database.
    """

    def __init__(self,
                 client: QdrantClient,
                 collection_name: str) -> None:
        """
        Construct a QdrantVectorStore object.

        To use you should have the ``qdrant-client`` package installed.

        :param client: the qdrant client object.
        :param collection_name: the name of the collection.
        """
        super().__init__()
        self._client = client
        self._collection_name = collection_name

    def add(self,
            point: Point,
            **kwargs: Any) -> str:
        structs = [point_to_point_struct(point)]
        self._logger.debug("Insert points: %s", structs)
        self._client.upsert(collection_name=self._collection_name,
                            points=structs,
                            **kwargs)
        return point.id

    def add_all(self,
                points: List[Point],
                **kwargs: Any) -> List[str]:
        structs = [point_to_point_struct(pt) for pt in points]
        self._logger.debug("Insert points: %s", structs)
        self._client.upsert(collection_name=self._collection_name,
                            points=structs,
                            **kwargs)
        return [p.id for p in points]

    def search(self,
               vector: Vector,
               limit: int,
               criterion: Optional[Criterion] = None,
               **kwargs: Any) -> List[Point]:
        query_filter = criterion_to_filter(criterion)
        self._logger.debug("Search: vector=%s, limit=%d, filter=%s",
                           vector, limit, query_filter)
        points = self._client.search(collection_name=self._collection_name,
                                     query_filter=query_filter,
                                     query_vector=vector,
                                     limit=limit,
                                     with_vectors=True,
                                     **kwargs)
        self._logger.debug("Search result: %s", points)
        return [scored_point_to_point(p) for p in points]


def point_to_point_struct(point: Point) -> models.PointStruct:
    """
    Converts a Point object into a qdrant PointStruct object.

    :param point: a Point object.
    :return: the converted PointStruct object.
    """
    if point.id is None:
        point.id = str(uuid.uuid4())
    return models.PointStruct(id=point.id,
                              vector=point.vector,
                              payload=point.metadata)


def scored_point_to_point(scored_point: models.ScoredPoint) -> Point:
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


def criterion_to_condition(criterion: Optional[Criterion])\
        -> Optional[models.Condition]:
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


def composed_criterion_to_filter(criterion: Optional[ComposedCriterion])\
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
