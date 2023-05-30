# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import Any

from .token_text_splitter import TokenTextSplitter
from ..llm.tokenizer import SpecialTokenSet, OpenAiTokenizer


class OpenAiTokenTextSplitter(TokenTextSplitter):
    """
    A text splitter which splits text by the number of tokens in OpenAI's models.
    """

    def __init__(self,
                 model: str,
                 allowed_special: SpecialTokenSet = None,
                 disallowed_special: SpecialTokenSet = "all",
                 **kwargs: Any) -> None:
        super().__init__(tokenizer=OpenAiTokenizer(model),
                         allowed_special=allowed_special,
                         disallowed_special=disallowed_special,
                         **kwargs)
