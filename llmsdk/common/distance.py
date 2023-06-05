# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import copy
from enum import Enum
from typing import List, Optional

import numpy as np

from .vector import Vector
from .point import Point


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
        Calculates the distance between two vectors with this distance metric.

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

    def accept_score(self, score: float, threshold: float) -> bool:
        """
        Tests whether the point with the specified score should be accepted
        in this distance metric with respect to the specified threshold.

        :param score: the score of the specified point.
        :param threshold: the threshold of the scores.
        :return: `True` if the point with the specified score should be accepted
            in this distance metric with respect to the specified threshold;
            `False` otherwise.
        """
        match self:
            case Distance.COSINE:
                # in the COSINE distance, the higher scores are better.
                return score >= threshold
            case Distance.DOT:
                # in the DOT distance, the higher scores are better
                return score >= threshold
            case Distance.EUCLID:
                # in the DOT distance, the lower scores are better
                return score <= threshold
            case _:
                raise ValueError(f"Unsupported distance: {self}")

    def sort(self, points: List[Point]) -> List[Point]:
        """
        Sorts a list of points by their scores with this distance metric.

        :param points: the list of points.
        :return: the sorted list of points.
        """
        match self:
            case Distance.COSINE:
                # in the COSINE distance, the higher scores are better.
                return sorted(points, key=lambda p: p.score, reverse=True)
            case Distance.DOT:
                # in the DOT distance, the higher scores are better
                return sorted(points, key=lambda p: p.score, reverse=True)
            case Distance.EUCLID:
                # in the DOT distance, the lower scores are better
                return sorted(points, key=lambda p: p.score, reverse=False)
            case _:
                raise ValueError(f"Unsupported distance: {self}")

    def calculate_score(self, query_vector: Vector, point: Point) -> Point:
        """
        Calculates the score of a point with respect to this distance metric.

        :param query_vector: the query vector.
        :param point: the specified point.
        :return: a new point same as the argument `point`, except the `score`
            field of the new point is set to the calculated score with respect
            to this distance metric.
        """
        score = self.between(point.vector, query_vector)
        return Point(id=point.id,
                     vector=copy.deepcopy(point.vector),
                     metadata=copy.deepcopy(point.metadata),
                     score=score)

    def calculate_scores(self, query_vector: Vector, points: List[Point]) -> List[Point]:
        """
        Calculates the score of a point with respect to this distance metric.

        :param query_vector: the query vector.
        :param points: the list of points.
        :return: a new list of points same as the argument `points`, except the
            `score` field of each point in the new list is set to the calculated
            score with respect to this distance metric.
        """
        return [self.calculate_score(query_vector, p) for p in points]

    def filter(self,
               points: List[Point],
               limit: int,
               score_threshold: Optional[float] = None) -> List[Point]:
        """
        Filters a list of points with respect to this distance metric.

        :param points: the list of points.
        :param limit: the maximum number of points to return.
        :param score_threshold: the threshold of the scores, or `None` if not
            specified.
        :return: the filtered list of points.
        """
        if score_threshold is None:
            return points[:limit]
        else:
            points = points[:limit]
            result = []
            for p in points:
                if self.accept_score(p.score, score_threshold):
                    result.append(p)
            return result
