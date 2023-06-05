# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from abc import ABC, abstractmethod

from ..common import Metadata


class Criterion(ABC):
    """The abstract interface of criterion criteria."""

    @abstractmethod
    def test(self, metadata: Metadata) -> bool:
        """
        Tests whether the specified metadata satisfies this criterion.

        :param metadata: the specified metadata.
        :return: `True` if the specified metadata satisfies this criterion;
            `False` otherwise.
        """
