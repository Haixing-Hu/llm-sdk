# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import Any, List

from ..llm.tokenizer.tokernizer import Tokenizer, SpecialTokenSet
from .text_splitter import TextSplitter


class TokenTextSplitter(TextSplitter):
    """
    A text splitter which splits text by the number of tokens.
    """

    def __init__(self,
                 tokenizer: Tokenizer,
                 allowed_special: SpecialTokenSet = None,
                 disallowed_special: SpecialTokenSet = "all",
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._tokenizer = tokenizer
        self._allowed_special = allowed_special
        self._disallowed_special = disallowed_special

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    def split_text(self, text: str) -> List[str]:
        splits = []
        tokens = self._tokenizer.encode(
            text,
            allowed_special=self._allowed_special,
            disallowed_special=self._disallowed_special
        )
        n = len(tokens)
        start = 0
        current = min(start + self._chunk_size, n)
        chunk = tokens[start:current]
        while start < n:
            splits.append(self._tokenizer.decode(chunk))
            start += self._chunk_size - self._chunk_overlap
            current = min(start + self._chunk_size, n)
            chunk = tokens[start:current]
        return splits
