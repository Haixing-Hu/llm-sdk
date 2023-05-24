# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from abc import ABC, abstractmethod


class IdGenerator(ABC):
    """
    The interface of generators generating unique identifiers.

    The subclasses implementing this interface should be thread-safe.
    """

    @abstractmethod
    def generate(self) -> str:
        """
        Generate a unique identifier.
        :return: a unique identifier.
        """
