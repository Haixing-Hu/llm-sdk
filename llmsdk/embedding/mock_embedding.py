# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from typing import Iterable, List

from ..common.vector import Vector
from ..util.math_utils import normalize_vector
from .embedding import Embedding


class MockEmbedding(Embedding):
    """
    A mock Embedding class used for testing.
    """

    VECTOR_DIMENSION: int = 10

    PRECISION: int = 7

    PREFIX: List[float] = [float(1.0)] * (VECTOR_DIMENSION - 1)

    def __init__(self) -> None:
        super().__init__(vector_dimension=MockEmbedding.VECTOR_DIMENSION,
                         use_cache=False)

    def _embed_impl(self, texts: Iterable[str]) -> Iterable[Vector]:
        """
        Return simple embeddings. Embeddings encode each text as its index.

        :param texts: the texts to be embedded.
        :return: the embedded vectors of each text.
        """
        vectors = (MockEmbedding.PREFIX + [float(i)] for i, _ in enumerate(texts))
        return (normalize_vector(v, MockEmbedding.PRECISION) for v in vectors)
