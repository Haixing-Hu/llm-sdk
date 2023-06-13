# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import Optional
import math

from ..common.vector import Vector


def normalize_vector(vector: Vector,
                     digits: Optional[int] = None) -> Vector:
    """
    Normalize a vector.

    :param vector: the vector to be normalized.
    :param digits: the number of digits to keep after the decimal point of each
        floating points in the normalized vector.
    :return: the normalized vector.
    """
    vector_length = math.sqrt(sum(x**2 for x in vector))
    normalized_vector = [x / vector_length for x in vector]
    if digits is None:
        return normalized_vector
    else:
        return [round(x, digits) for x in normalized_vector]


def euclid_distance(v1: Vector, v2: Vector) -> float:
    """
    Calculates the Euclid distance between two vectors.

    :param v1: the first vector.
    :param v2: the second vector.
    :return: the Euclid distance between the two vectors.
    """
    return math.sqrt(sum((x1 - x2)**2 for x1, x2 in zip(v1, v2)))


def dot_distance(v1: Vector, v2: Vector) -> float:
    """
    Calculates the dot distance between two vectors.

    :param v1: the first vector.
    :param v2: the second vector.
    :return: the dot distance between the two vectors.
    """
    return sum(x1 * x2 for x1, x2 in zip(v1, v2))


def cosine_distance(v1: Vector, v2: Vector) -> float:
    """
    Calculates the cosine distance between two vectors.

    :param v1: the first vector.
    :param v2: the second vector.
    :return: the cosine distance between the two vectors.
    """
    v1_length = math.sqrt(sum(x**2 for x in v1))
    v2_length = math.sqrt(sum(x**2 for x in v2))
    return dot_distance(v1, v2) / (v1_length * v2_length)
