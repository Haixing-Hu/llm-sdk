# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import List

from ..common import Vector
from ..util.common_utils import normalize_vector
from .embedding import Embedding


class MockEmbedding(Embedding):
    """
    A mock Embedding class used for testing.
    """

    PRECISION: int = 7

    def __init__(self) -> None:
        super().__init__(output_dimensions=10)

    def embed_texts(self, texts: List[str]) -> List[Vector]:
        """
        Return simple embeddings. Embeddings encode each text as its index.

        :param texts: the list of texts.
        :return: the list of embedded vectors of each text.
        """
        vectors = [[float(1.0)] * 9 + [float(i)] for i in range(len(texts))]
        return [normalize_vector(v, MockEmbedding.PRECISION) for v in vectors]
