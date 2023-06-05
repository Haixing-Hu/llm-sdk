# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import List, Optional

import openai

from ..common import Vector
from ..llm.tokenizer import Tokenizer, OpenAiTokenizer
from ..util.openai_utils import (
    get_embedding_output_dimensions,
    check_model_compatibility,
    call_with_retries,
    init_openai,
)
from .embedding import Embedding


class OpenAiEmbedding(Embedding):
    """
    The embedding model using the OpenAI API.
    """

    DEFAULT_MODEL = "text-embedding-ada-002"

    DEFAULT_BATCH_SIZE = 1000

    def __init__(self,
                 model: str = DEFAULT_MODEL,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 api_key: Optional[str] = None,
                 use_proxy: Optional[bool] = None) -> None:
        super().__init__(output_dimensions=get_embedding_output_dimensions(model))
        check_model_compatibility(model=model, endpoint="embeddings")
        self._model = model
        self._batch_size = batch_size
        self._tokenizer = OpenAiTokenizer(model)
        init_openai(api_key=api_key, use_proxy=use_proxy)

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    # def embed_texts(self, texts: List[str]) -> List[Vector]:
    #     # split all documents into list of chunked token lists
    #     self._logger.debug("Embedding texts: %s", texts)
    #     all_token_lists = []
    #     indices = []
    #     for i, text in enumerate(texts):
    #         token_list = get_chunked_tokens(self._model, self._tokenizer, text)
    #         # append the token list to the end of all_token_lists
    #         all_token_lists += token_list
    #         # remember the document index of each token list
    #         indices += [i] * len(token_list)
    #
    #     # batch embedding all token lists
    #     embeddings = []
    #     for i in range(0, len(all_token_lists), self._batch_size):
    #         input = all_token_lists[i:i+self._batch_size]
    #         self._logger.debug("Embed %d chunks with OpenAI: %s", len(input), input)
    #         response = call_with_retries(openai_api=openai.Embedding.create,
    #                                      model=self._model,
    #                                      input=input,
    #                                      encoding_format="float")
    #         embedding = [r["embedding"] for r in response["data"]]
    #         self._logger.debug("Gets the embedded vectors of the chunks: %s", embedding)
    #         embeddings += embedding
    #
    #     # collect the embedding vectors of each document
    #     n = len(texts)
    #     # NOTE: we CANNOT use [[]] * n to initialize the empty list of list,
    #     #   since Python will use the same empty list object to fill the n elements
    #     #   of [[]] * n
    #     vectors: List[List[List[float]]] = [[] for _ in range(n)]
    #     lengths: List[List[int]] = [[] for _ in range(n)]
    #     for i in range(len(embeddings)):
    #         embedding = embeddings[i]
    #         index = indices[i]
    #         vectors[index].append(embedding)
    #         lengths[index].append(len(embedding))
    #
    #     # join the embedding vectors of each document
    #     result = []
    #     for i in range(n):
    #         average = np.average(vectors[i],
    #                              axis=0,
    #                              weights=lengths[i])
    #         point = (average / np.linalg.norm(average)).tolist()
    #         result.append(point)
    #     return result

    def embed_texts(self, texts: List[str]) -> List[Vector]:
        self._logger.debug("Embedding texts: %s", texts)
        tokens_list = []
        for i, text in enumerate(texts):
            tokens = self._tokenizer.encode(text)
            tokens_list.append(tokens)
        # batch embedding all token lists
        result = []
        for i in range(0, len(tokens_list), self._batch_size):
            input_list = tokens_list[i:i+self._batch_size]
            self._logger.debug("Embed %d chunks with OpenAI: %s", len(input_list), input_list)
            response = call_with_retries(openai_api=openai.Embedding.create,
                                         model=self._model,
                                         input=input_list)
            embedding_list = [r["embedding"] for r in response["data"]]
            self._logger.debug("Gets the embedded vectors of the chunks: %s", embedding_list)
            result.extend(embedding_list)
        return result
