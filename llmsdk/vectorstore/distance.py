# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from enum import Enum


class Distance(Enum):
    """The enumeration of distances bewteen points."""

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
