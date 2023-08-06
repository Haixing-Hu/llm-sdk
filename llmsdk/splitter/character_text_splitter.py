# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from typing import List, Any

from .text_splitter import TextSplitter
from .text_splitter_utils import combine_splits


class CharacterTextSplitter(TextSplitter):
    """
    A TextSplitter splits texts with the specified separator.
    """

    def __init__(self,
                 separator: str = "\n\n",
                 **kwargs: Any) -> None:
        """
        Creates a splitter which splits text with the specified separator.

        :param separator: the specified separator.
        :param kwargs: other arguments passed to the constructor of the super class.
        """
        super().__init__(**kwargs)
        if separator is None:
            raise ValueError(f"Invalid separator: {separator}")
        self._separator = separator

    @property
    def separator(self) -> str:
        return self._separator

    def split_text(self, text: str) -> List[str]:
        if len(self._separator) > 0:
            splits = text.split(self._separator)
        else:
            splits = list(text)
        return combine_splits(splits=splits,
                              separator=self._separator,
                              chunk_size=self._chunk_size,
                              chunk_overlap=self._chunk_overlap,
                              length_function=self._length_function)
