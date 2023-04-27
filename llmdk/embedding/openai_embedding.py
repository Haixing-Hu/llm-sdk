# ##############################################################################
#                                                                              #
#    Copyright (c) 2023.  Haixing Hu.                                          #
#                                                                              #
#    All rights reserved.                                                      #
#                                                                              #
# ##############################################################################
import logging
import openai
import numpy as np
from .embedding import Embedding
from llmdk.util.openai_utils import (
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
                 api_key: str = None,
                 use_proxy: bool = None) -> None:
        super().__init__()
        self._model = model
        self._batch_size = batch_size
        self._logger = logging.getLogger(self.__class__.__name__)
        check_model_compatibility(model=model, endpoint="embeddings")
        init_openai(api_key=api_key,
                    use_proxy=use_proxy)

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        # split all documents into list of chunked token lists
        all_token_lists = []
        indices = []
        for i, text in enumerate(documents):
            token_list = get_chunked_tokens(self._model, text)
            # append the token list to the end of all_token_lists
            all_token_lists += token_list
            # remember the document index of each token list
            indices += [i] * len(token_list)

        # batch embedding all token lists
        all_embeddings = []
        for i in range(0, len(all_token_lists), self._batch_size):
            embedding_input = all_token_lists[i:i+self._batch_size]
            response = call_with_retries(openai_api=openai.Embedding.create,
                                         model=self._model,
                                         input=embedding_input)
            all_embeddings += [r["embedding"] for r in response["data"]]

        # collect the embedding vectors of each document
        n = len(documents)
        vectors: list[list[list[float]]] = [[]] * n
        lengths: list[list[int]] = [[]] * n
        for i in range(len(all_embeddings)):
            embedding = all_embeddings[i]
            doc_index = indices[i]
            vectors[doc_index].append(embedding)
            lengths[doc_index].append(len(embedding))

        # join the embedding vectors of each document
        result = []
        for i in range(n):
            average = np.average(vectors[i],
                                 axis=0,
                                 weights=lengths[i])
            result.append((average / np.linalg.norm(average)).tolist())
        return result

    def embed_query(self, query: str) -> list[float]:
        vector_lists = self.embed_documents([query])
        return vector_lists[0]