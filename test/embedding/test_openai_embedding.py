# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import unittest
import logging

import openai
from llmsdk.embedding import OpenAiEmbedding


class TestOpenAiEmbedding(unittest.TestCase):

    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)
        openai.debug = True
        openai.log = "debug"

    def test_embed_query(self) -> None:
        embedding = OpenAiEmbedding()
        vectors = embedding.embed_query("Hello world!")
        print(vectors)

    def test_embed_documents(self) -> None:
        pass


if __name__ == '__main__':
    unittest.main()
