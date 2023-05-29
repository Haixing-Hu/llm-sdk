# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import Any, List, Optional

from .text_splitter import TextSplitter
from .text_splitter_utils import combine_splits


class RecursiveCharacterTextSplitter(TextSplitter):
    """
    A text splitter recursively tries to split by different characters
    to find one that works.
    """

    def __init__(self,
                 separators: Optional[List[str]] = None,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> List[str]:
        result = []
        # Get appropriate separator to use
        separator = self._separators[-1]
        for sep in self._separators:
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                break
        # Now that we have the separator, split the text
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)
        # Now go merging things, recursively splitting longer texts.
        good_splits = []
        for split in splits:
            if self._length_function(split) < self._chunk_size:
                good_splits.append(split)
            else:
                if good_splits:
                    merged_text = combine_splits(splits=good_splits,
                                                 separator=separator,
                                                 chunk_size=self._chunk_size,
                                                 chunk_overlap=self._chunk_overlap,
                                                 length_function=self._length_function)
                    result.extend(merged_text)
                    good_splits = []
                other_info = self.split_text(split)  # recursive call
                result.extend(other_info)
        if len(good_splits) > 0:
            merged_text = combine_splits(splits=good_splits,
                                         separator=separator,
                                         chunk_size=self._chunk_size,
                                         chunk_overlap=self._chunk_overlap,
                                         length_function=self._length_function)
            result.extend(merged_text)
        return result
