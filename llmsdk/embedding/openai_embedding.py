# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import logging
from typing import List, Optional

import numpy as np
import openai

from .embedding import Embedding
from ..common import Vector
from ..util.openai_utils import (
    check_model_compatibility,
    call_with_retries,
    get_chunked_tokens,
    init_openai,
)

DEFAULT_MODEL = "text-embedding-ada-002"
DEFAULT_BATCH_SIZE = 1000


class OpenAiEmbedding(Embedding):
    """
    The embedding model using the OpenAI API.
    """
    def __init__(self,
                 model: str = DEFAULT_MODEL,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 api_key: Optional[str] = None,
                 use_proxy: Optional[bool] = None) -> None:
        super().__init__()
        self._model = model
        self._batch_size = batch_size
        check_model_compatibility(model=model, endpoint="embeddings")
        init_openai(api_key=api_key,
                    use_proxy=use_proxy)

    def _embed_texts(self, texts: List[str]) -> List[Vector]:
        # split all documents into list of chunked token lists
        all_token_lists = []
        indices = []
        for i, text in enumerate(texts):
            token_list = get_chunked_tokens(self._model, text)
            # append the token list to the end of all_token_lists
            all_token_lists += token_list
            # remember the document index of each token list
            indices += [i] * len(token_list)

        # batch embedding all token lists
        embeddings = []
        for i in range(0, len(all_token_lists), self._batch_size):
            input = all_token_lists[i:i+self._batch_size]
            response = call_with_retries(openai_api=openai.Embedding.create,
                                         model=self._model,
                                         input=input)
            embeddings += [r["embedding"] for r in response["data"]]

        # collect the embedding vectors of each document
        n = len(texts)
        vectors: List[List[List[float]]] = [[]] * n
        lengths: List[List[int]] = [[]] * n
        for i in range(len(embeddings)):
            embedding = embeddings[i]
            index = indices[i]
            vectors[index].append(embedding)
            lengths[index].append(len(embedding))

        # join the embedding vectors of each document
        result = []
        for i in range(n):
            average = np.average(vectors[i],
                                 axis=0,
                                 weights=lengths[i])
            point = (average / np.linalg.norm(average)).tolist()
            result.append(point)
        return result
