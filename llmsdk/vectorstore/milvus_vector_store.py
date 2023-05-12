# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import Dict, Optional, Any, List
import uuid

from pymilvus import Collection, DataType, connections, Index

from .vector_store import VectorStore
from ..common import Vector, Point
from ..criterion import Criterion, SimpleCriterion, ComposedCriterion, Operator, Relation

DEFAULT_INDEX_PARAMS = {
    "IVF_FLAT": {"nprobe": 10},
    "IVF_SQ8": {"nprobe": 10},
    "IVF_PQ": {"nprobe": 10},
    "HNSW": {"ef": 10},
    "RHNSW_FLAT": {"ef": 10},
    "RHNSW_SQ": {"ef": 10},
    "RHNSW_PQ": {"ef": 10},
    "IVF_HNSW": {"nprobe": 10, "ef": 10},
    "ANNOY": {"search_k": 10},
}


class MilvusVectorStore(VectorStore):
    """
    The vector store based on the Milvus vector database.
    """

    def __init__(self,
                 collection_name: str,
                 id_field: Optional[str] = None,
                 vector_field: Optional[str] = None,
                 connection_args: Optional[Dict] = None,
                 auto_close_connection: bool = True) -> None:
        """
        Construct a vector store based on a collection of a Milvus vector
        database.

        :param collection_name: the name of the collection in the vector database.
        :param id_field: the name of ID field in the collection.
        :param vector_field: the name of vector field in the collection.
        :param connection_args: the arguments for the database connection.
        :param auto_close_connection: indicate whether to close the connection
            automatically while closing this vector store.
        """
        super().__init__()
        # Connecting to Milvus instance
        if connection_args is None:
            self._connection_args = None
            self._connection_alias = "default"
        else:
            self._connection_args = connection_args
            self._connection_alias = connection_args.get("alias", "default")
            if not connections.has_connection(self._connection_alias):
                connections.connect(**connection_args)
        self._collection = Collection(name=collection_name,
                                      using=self._connection_alias)
        self._auto_id = self._collection.schema.auto_id
        self._auto_close_connection = auto_close_connection
        self._id_field = id_field
        self._vector_field = vector_field
        self._fields = []
        # grab the fields for the existing collection.
        for f in self._collection.schema.fields:
            self._fields.append(f.name)
            if f.is_primary and self._id_field is None:
                self._id_field = f.name
            elif f.dtype == DataType.FLOAT_VECTOR and self._vector_field is None:
                self._vector_field = f.name
        if self._id_field is None:
            raise ValueError(f"No primary field in the collection '{collection_name}'.")
        if self._vector_field is None:
            raise ValueError(f"No float vector field in the collection '{collection_name}'.")
        if self._id_field not in self._fields:
            raise ValueError(f"The ID field '{self._id_field}' is not in the "
                             f"collection '{collection_name}'.")
        if self._vector_field not in self._fields:
            raise ValueError(f"The vector field '{self._vector_field}' is not in "
                             f"the collection '{collection_name}'.")
        # find the index for the vector field
        self._vector_index = None
        for index in self._collection.indexes:
            if index.field_name == self._vector_field:
                self._vector_index = index
        if self._vector_index is None:
            raise ValueError(f"No index for the vector field '{self._vector_field}' "
                             f"in the collection '{collection_name}'.")
        # exclude the ID field and vector field from the metadata fields
        self._metadata_fields = self._fields.copy()
        self._metadata_fields.remove(self._id_field)
        self._metadata_fields.remove(self._vector_field)
        # load the collection
        self._collection.load()

    def add(self,
            point: Point,
            **kwargs: Any) -> str:
        ids = self.add_all([point], **kwargs)
        return ids[0]

    def add_all(self,
                points: List[Point],
                **kwargs: Any) -> List[str]:
        data: List[List[Any]] = [] * len(self._fields)
        for p in points:
            if not self._auto_id and p.id is None:
                p.id = str(uuid.uuid4())
            for i, field in enumerate(self._fields):
                if field == self._id_field:
                    data[i].append(p.id)
                elif field == self._vector_field:
                    data[i].append(p.vector)
                else:
                    if p.metadata is None or field not in p.metadata.keys():
                        value = None
                    else:
                        value = p.metadata[field]
                    data[i].append(value)
        self._logger.debug("Insert data: %s", data)
        result = self._collection.insert(data=data, **kwargs)
        self._collection.flush()
        # set the automatically generated IDs for points
        if self._auto_id:
            for i, value in enumerate(result.primary_keys):
                points[i].id = value
        return [p.id for p in points]

    def search(self,
               vector: Vector,
               limit: int,
               criterion: Optional[Criterion] = None,
               **kwargs: Any) -> List[Point]:
        params = {"metric_type": self._vector_index.params["metric_type"]}
        index_type = self._vector_index.params["index_type"]
        if index_type in DEFAULT_INDEX_PARAMS:
            params["params"] = DEFAULT_INDEX_PARAMS[index_type]
        else:
            params["params"] = self._vector_index.params["params"]
        expr = criterion_to_expr(criterion)
        results = self._collection.search(data=[vector],
                                          anns_field=self._vector_field,
                                          param=params,
                                          limit=limit,
                                          expr=expr,
                                          output_fields=self._fields,
                                          **kwargs)
        points = []
        for r in results[0]:
            vector = r.entity.get(self._vector_field)
            metadata = {}
            for f in self._metadata_fields:
                v = r.entity.get(f)
                if v is not None:
                    metadata[f] = v
            point = Point(id=r.id,
                          vector=vector,
                          metadata=metadata,
                          score=r.distance)
            points.append(point)
        return points

    def close(self) -> None:
        self._collection.release()
        if self._auto_close_connection:
            connections.disconnect(self._connection_alias)


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
