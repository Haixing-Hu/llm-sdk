# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from dataclasses import dataclass, field
from typing import List

from ..common.distance import Distance
from .payload_schema import PayloadSchema


@dataclass(frozen=True)
class CollectionInfo:
    """
    The class of object storing the information about a collection.
    """

    name: str
    """
    The name of the collection.
    """

    size: int
    """
    The number of points stored in the collection.
    """

    vector_dimension: int
    """
    The the dimension of vectors stored in the collection.
    """

    distance: Distance = Distance.COSINE
    """
    The distance used to estimate the similarity of vectors with each other.
    """

    payload_schemas: List[PayloadSchema] = field(default_factory=list)
    """
    The list of schemas of payloads of the points in the collection.
    """
