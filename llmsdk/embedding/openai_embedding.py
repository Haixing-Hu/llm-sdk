# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from typing import Any, List, Optional

from ..common.vector import Vector
from ..llm.tokenizer.tokernizer import Tokenizer
from ..llm.tokenizer.openai_tokenizer import OpenAiTokenizer
from ..util.openai_utils import (
    get_model_tokens,
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
                 use_proxy: Optional[bool] = None,
                 **kwargs: Any) -> None:
        """
        Creates an OpenAiEmbedding object.

        :param model: the name of the OpenAI model to be used.
        :param batch_size: the batch size of the OpenAI API.
        :param api_key: the API key of the OpenAI API.
        :param use_proxy: indicates whether to use the proxy to access the
            OpenAI API.
        :param kwargs: the extra arguments passed to the constructor of the
            base class.
        """
        super().__init__(
            vector_dimension=get_embedding_output_dimensions(model),
            **kwargs,
        )
        try:
            import openai
        except ImportError:
            raise ImportError("Openai Python package is not installed, please "
                              "install it with `pip install openai`.")
        check_model_compatibility(model=model, endpoint="embeddings")
        self._model = model
        self._model_tokens = get_model_tokens(model)
        self._batch_size = batch_size
        self._tokenizer = OpenAiTokenizer(model)
        self._api = openai.Embedding.create
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
    #         response = call_with_retries(openai_api=self._api,
    #                                      model=self._model,
    #                                      input=input)
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

    def _embed_impl(self, texts: List[str]) -> List[Vector]:
        tokens_list = []
        for i, text in enumerate(texts):
            tokens = self._tokenizer.encode(text)
            if len(tokens) > self._model_tokens:
                raise ValueError(f"The text is too long: {len(tokens)} tokens, "
                                 f"but the OpenAI model {self._model} only "
                                 f"supports {self._model_tokens} tokens: {text}")
            tokens_list.append(tokens)
        # batch embedding all token lists
        result = []
        for i in range(0, len(tokens_list), self._batch_size):
            input_list = tokens_list[i:i+self._batch_size]
            self._logger.debug("Embed %d chunks with OpenAI: %s",
                               len(input_list), input_list)
            response = call_with_retries(openai_api=self._api,
                                         model=self._model,
                                         input=input_list)
            embedding_list = [r["embedding"] for r in response["data"]]
            self._logger.debug("Gets the embedded vectors of the chunks: %s",
                               embedding_list)
            result.extend(embedding_list)
        return result
