# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from logging import Logger, getLogger


class WithLoggerMixin:
    """
    A mixin class that provides the logger function.
    """

    def __init__(self, **kwargs) -> None:
        """
        Creates a new WithLoggerMixin object.
        """
        super().__init__()      # This MUST be called
        self._logger = getLogger(self.__class__.__name__)

    @property
    def logger(self) -> Logger:
        """
        Gets the logger of this object.

        :return: the logger of this object.
        """
        return self._logger

    def set_logging_level(self, level: int | str) -> None:
        """
        Sets the logging level of this object.

        :param level: the logging level to be set.
        """
        self._logger.setLevel(level)
