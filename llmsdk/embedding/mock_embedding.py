# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import List

from ..common import Vector
from .embedding import Embedding


class MockEmbedding(Embedding):
    """
    A mock Embedding class used for testing.
    """

    def __init__(self) -> None:
        super().__init__(output_dimensions=10)

    def _embed_texts(self, texts: List[str]) -> List[Vector]:
        """
        Return simple embeddings. Embeddings encode each text as its index.

        :param texts: the list of texts.
        :return: the list of embedded vectors of each text.
        """
        return [[float(1.0)] * 9 + [float(i)] for i in range(len(texts))]
