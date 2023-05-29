# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import Any, List

from .text_splitter import TextSplitter
from .text_splitter_utils import combine_splits


class NltkSentenceTextSplitter(TextSplitter):
    """
    A text splitter which splits text that looks at sentences using NLTK.
    """

    def __init__(self,
                 separator: str = "\n\n",
                 language: str = "english",
                 **kwargs: Any):
        super().__init__(**kwargs)
        try:
            from nltk.tokenize import sent_tokenize
        except ImportError:
            raise ImportError(
                "NLTK is not installed, please install it with `pip install nltk`."
            )
        self._tokenize = sent_tokenize
        self._separator = separator
        self._language = language

    def split_text(self, text: str) -> List[str]:
        splits = self._tokenize(text, language=self._language)
        return combine_splits(splits=splits,
                              separator=self._separator,
                              chunk_size=self._chunk_size,
                              chunk_overlap=self._chunk_overlap,
                              length_function=self._length_function)
