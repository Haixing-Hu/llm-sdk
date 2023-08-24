# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from typing import Any, List

from .text_splitter import TextSplitter
from .text_splitter_utils import combine_splits


class SpacySentenceTextSplitter(TextSplitter):
    """
    A text splitter which splits text that looks at sentences using Spacy.
    """

    def __init__(self,
                 separator: str = "\n\n", *,
                 pipeline: str = "en_core_web_sm",
                 **kwargs: Any):
        super().__init__(**kwargs)
        try:
            import spacy
        except ImportError:
            raise ImportError(
                "Spacy is not installed, please install it with `pip install spacy`."
            )
        self._tokenize = spacy.load(pipeline)
        self._separator = separator
        self._pipeline = pipeline

    def split_text(self, text: str) -> List[str]:
        splits = (str(s) for s in self._tokenize(text).sents)
        return combine_splits(splits=splits,
                              separator=self._separator,
                              chunk_size=self._chunk_size,
                              chunk_overlap=self._chunk_overlap,
                              length_function=self._length_function)
