# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from typing import Iterable, Any, Optional

from tqdm import tqdm


class WithProgressMixin:
    """
    A mixin class that provides the progress bar function.
    """

    def __init__(self, *,
                 show_progress: bool = False,
                 show_progress_threshold: int = 10) -> None:
        """
        Creates a new ShowProgressMixin object.

        :param show_progress: indicates whether to show the progress of adding
            records.
        :param show_progress_threshold: the minimum number of items to show
            the progress.
        """
        super().__init__()      # This MUST be called
        self._show_progress = show_progress
        self._show_progress_threshold = show_progress_threshold

    @property
    def show_progress(self) -> bool:
        return self._show_progress

    @show_progress.setter
    def show_progress(self, value: bool) -> None:
        self._show_progress = value

    @property
    def show_progress_threshold(self) -> int:
        return self._show_progress_threshold

    @show_progress_threshold.setter
    def show_progress_threshold(self, value: int) -> None:
        self._show_progress_threshold = value

    def _get_iterable(self,
                      iterable: Iterable[Any],
                      desc: Optional[str] = None,
                      total: Optional[int] = None) -> Iterable[Any]:
        """
        Shows a progress bar if the number of items in the iterable is greater
        than the threshold to show progress.

        :param iterable: the iterable to be wrapped.
        :param desc: the description of the progress bar.
        :param total: the total number of items in the iterable.
        :return: the wrapped iterable.
        """
        if self._show_progress:
            if total is None:
                if hasattr(iterable, '__len__'):
                    total = len(iterable)
            if total is None:
                return tqdm(iterable, desc=desc)
            elif total >= self._show_progress_threshold:
                return tqdm(iterable, desc=desc, total=total)
        return iterable
