# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from typing import Optional, List

from ..common.data_type import DataType
from ..common.distance import Distance
from ..criterion.operator import Operator
from ..criterion.relation import Relation
from ..criterion.criterion import Criterion
from ..criterion.simple_criterion import SimpleCriterion
from ..criterion.composed_criterion import ComposedCriterion
from .payload_schema import PayloadSchema


DEFAULT_ID_FIELD_NAME = "__id__"
"""
The default field name of the primary ID field.
"""

DEFAULT_VECTOR_FIELD_NAME = "__vector__"
"""
The default field name of the floating vector field.
"""

DEFAULT_VECTOR_INDEX_TYPE = "HNSW"
"""
The default vector index type.
"""

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
"""
The default index params for different index types.
"""

IMPORT_MILVUS_ERROR_MESSAGE = """Milvus is not installed, 
please install it with `pip install pymilvus`."""


def to_milvus_distance(distance: Distance) -> str:
    """
    Converts the vector distance used in this library into the vector distance
    used in the Milvus.

    :param distance: the vector distance used in this library.
    :return: the corresponding vector distance used in the Milvus.
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
    Converts the vector distance used in the Milvus into the vector distance
    used in this library.

    :param distance: the vector distance used in the Milvus.
    :return: the corresponding vector distance used in this library.
    """
    match distance:
        case "IP":
            return Distance.COSINE
        case "L2":
            return Distance.EUCLID
        case _:
            raise ValueError(f"Unsupported distance type: {distance}")


def to_milvus_type(data_type: DataType):
    """
    Converts the data type used in this library into the data type used in the
    Milvus.

    :param data_type: the data type used in this library.
    :return: the corresponding data type used in the Milvus.
    """
    try:
        import pymilvus
    except ImportError:
        raise ImportError(IMPORT_MILVUS_ERROR_MESSAGE)
    match data_type:
        case DataType.INT:
            return pymilvus.DataType.INT64
        case DataType.FLOAT:
            return pymilvus.DataType.FLOAT
        case DataType.STRING:
            return pymilvus.DataType.STRING
        case _:
            raise ValueError(f"Unsupported data type: {data_type}")


def to_local_type(data_type) -> DataType:
    """
    Converts the data type used in the Milvus into the data type used in this
    library.

    :param data_type: the data type used in the Milvus.
    :return: the corresponding data type used in this library.
    """
    try:
        import pymilvus
    except ImportError:
        raise ImportError(IMPORT_MILVUS_ERROR_MESSAGE)
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


def to_milvus_field_schema(schema: PayloadSchema):
    """
    Converts a local payload schema to a FieldSchema in pymilvus.

    :param schema: the local payload schema.
    :return: the corresponding FieldSchema in pymilvus.
    """
    try:
        import pymilvus
    except ImportError:
        raise ImportError(IMPORT_MILVUS_ERROR_MESSAGE)
    return pymilvus.FieldSchema(name=schema.name,
                                dtype=to_milvus_type(schema.type))


def get_vector_field(collection, field_name: Optional[str] = None):
    """
    Gets the schema of the vector field of the specified Milvus collection.

    :param collection: a specified Milvus collection.
    :param field_name: the optional name of the vector field.
    :return: the schema of the vector field of the specified Milvus collection.
    """
    try:
        import pymilvus
    except ImportError:
        raise ImportError(IMPORT_MILVUS_ERROR_MESSAGE)
    fields = {f.name: f for f in collection.schema.fields}
    # get the vector field of the specified name
    if field_name is not None:
        if field_name in fields:
            field = fields.get(field_name)
            if field.dtype != pymilvus.DataType.FLOAT_VECTOR:
                raise ValueError(f"The field '{field_name}' in the collection "
                                 f"'{collection.name}' is not a float vector.")
            return field
        else:
            raise ValueError(f"Cannot find the field '{field_name}' in the "
                             f"collection '{collection.name}'")
    # get the vector field of the default name
    if DEFAULT_VECTOR_FIELD_NAME in fields:
        field = fields.get(DEFAULT_VECTOR_FIELD_NAME)
        if field.dtype != pymilvus.DataType.FLOAT_VECTOR:
            raise ValueError(f"The field '{DEFAULT_VECTOR_FIELD_NAME}' in the "
                             f"collection '{collection.name}' is not a float vector.")
        return field
    # find the first vector field in the collection
    for field in collection.schema.fields:
        if field.dtype == pymilvus.DataType.FLOAT_VECTOR:
            return field
    raise ValueError(f"No vector field found in the collection '{collection.name}'.")


def get_id_field(collection, field_name: Optional[str] = None):
    """
    Gets the schema of the ID field of the specified Milvus collection.

    :param collection: a specified Milvus collection.
    :param field_name: the optional name of the ID field.
    :return: the schema of the ID field of the specified Milvus collection.
    """
    fields = {f.name: f for f in collection.schema.fields}
    # get the vector field of the specified name
    if field_name is not None:
        if field_name in fields:
            field = fields.get(field_name)
            if not field.is_primary:
                raise ValueError(f"The field '{field_name}' in the collection "
                                 f"'{collection.name}' is not a primary ID field.")
            return field
        else:
            raise ValueError(f"Cannot find the field '{field_name}' in the "
                             f"collection '{collection.name}'")
    # get the vector field of the default name
    if DEFAULT_ID_FIELD_NAME in fields:
        field = fields.get(DEFAULT_ID_FIELD_NAME)
        if not field.is_primary:
            raise ValueError(f"The field '{DEFAULT_ID_FIELD_NAME}' in the "
                             f"collection '{collection.name}' is not a primary "
                             f"ID field.")
        return field
    # find the first vector field in the collection
    for field in collection.schema.fields:
        if field.is_primary:
            return field
    raise ValueError(f"No primary ID field found in the collection '{collection.name}'.")


def get_index(collection, field_name: str):
    """
    Gets the index of the specified field of the specified Milvus collection.

    :param collection: a specified Milvus collection.
    :param field_name: the name of the specified field.
    :return: the index of the specified field of the specified Milvus collection.
    """
    for index in collection.indexes:
        if index.field_name == field_name:
            return index
    raise ValueError(f"Cannot find the index of the field '{field_name}' in the "
                     f"collection '{collection.name}'")


def get_payload_schemas(collection, id_field, vector_field) -> List[PayloadSchema]:
    """
    Gets the list of payload schemas of the specified collection.

    :param collection: the specified collection.
    :param id_field: the ID field of the collection.
    :param vector_field: the vector field of the collection.
    :return: the list of payload schemas of the specified collection.
    """
    result = []
    for field in collection.schema.fields:
        if field.name != id_field.name and field.name != vector_field.name:
            schema = PayloadSchema(field.name, to_local_type(field.dtype))
            result.append(schema)
    return result


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
