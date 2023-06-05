# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from enum import Enum

import numpy as np

from .vector import Vector


class Distance(Enum):
    """The enumeration of distances between points."""

    EUCLID = "Euclid"
    """
    The Euclid distance between two points.
    
    See: https://en.wikipedia.org/wiki/Euclidean_distance
    """

    COSINE = "Cosine"
    """
    In data analysis, cosine similarity is a measure of similarity between two 
    non-zero vectors defined in an inner product space. Cosine similarity is the 
    cosine of the angle between the vectors; that is, it is the dot product of 
    the vectors divided by the product of their lengths. It follows that the 
    cosine similarity does not depend on the magnitudes of the vectors, but 
    only on their angle.
    
    See: https://en.wikipedia.org/wiki/Cosine_similarity
    """

    DOT = "Dot"
    """
    The dot product (scalar product), or inner product between two points. 
    
    Algebraically, the dot product is the sum of the products of the 
    corresponding entries of the two sequences of numbers. Geometrically, it is
    the product of the Euclidean magnitudes of the two vectors and the cosine 
    of the angle between them. These definitions are equivalent when using 
    Cartesian coordinates. 
    
    In other words, if you use inner product to calculate embeddings similarities, 
    you must normalize your embeddings. After normalization, the inner product 
    equals cosine similarity.
        
    See: https://en.wikipedia.org/wiki/Dot_product
    """

    def between(self, v1: Vector, v2: Vector) -> float:
        """
        Calculates the distance between two vectors.

        :param v1: the first vector.
        :param v2: the second vector.
        :return: the distance between two vectors.
        """
        match self:
            case Distance.COSINE:
                v1 = np.array(v1)
                v2 = np.array(v2)
                dot_product = np.dot(v1, v2)
                v1_length = np.linalg.norm(v1)
                v2_length = np.linalg.norm(v2)
                return dot_product / (v1_length * v2_length)
            case Distance.DOT:
                v1 = np.array(v1)
                v2 = np.array(v2)
                return float(np.dot(v1, v2))
            case Distance.EUCLID:
                v1 = np.array(v1)
                v2 = np.array(v2)
                diff = v1 - v2
                squared_diff = diff ** 2
                sum_squared_diff = np.sum(squared_diff)
                return np.sqrt(sum_squared_diff)
            case _:
                raise ValueError(f"Unsupported distance: {self}")
