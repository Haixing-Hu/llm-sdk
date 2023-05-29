# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import List, Any

from .text_splitter import TextSplitter


class SeparatorTextSplitter(TextSplitter):
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
        if separator is None or len(separator) == 0:
            raise ValueError(f"Invalid separator: {separator}")
        self._separator = separator

    @property
    def separator(self) -> str:
        return self._separator

    def split_text(self, text: str) -> List[str]:
        return text.split(self._separator)

    def join_texts(self, texts: List[str]) -> str:
        pass
