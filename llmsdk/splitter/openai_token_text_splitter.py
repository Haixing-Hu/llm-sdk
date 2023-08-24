# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from typing import Any

from ..llm.tokenizer.tokernizer import SpecialTokenSet
from ..llm.tokenizer.openai_tokenizer import OpenAiTokenizer
from ..embedding.openai_embedding import OpenAiEmbedding
from .token_text_splitter import TokenTextSplitter


class OpenAiTokenTextSplitter(TokenTextSplitter):
    """
    A text splitter which splits text by the number of tokens in OpenAI's models.
    """

    def __init__(self,
                 model: str = OpenAiEmbedding.DEFAULT_MODEL, *,
                 allowed_special: SpecialTokenSet = None,
                 disallowed_special: SpecialTokenSet = "all",
                 **kwargs: Any) -> None:
        """
        Constructs a OpenAiTokenTextSplitter.

        :param model: the name of the OpenAI's model.
        :param allowed_special: the allowed special tokens.
        :param disallowed_special: the disallowed special tokens. Default value
            is "all", which means all special tokens are disallowed.
        :param kwargs: the extra arguments for the TokenTextSplitter.
        """
        super().__init__(tokenizer=OpenAiTokenizer(model),
                         allowed_special=allowed_special,
                         disallowed_special=disallowed_special,
                         **kwargs)
