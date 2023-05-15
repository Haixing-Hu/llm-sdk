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
from llmsdk.common import Document


class TestOpenAiEmbedding(unittest.TestCase):

    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)
        openai.debug = True
        openai.log = "debug"

    def test_embed_query(self) -> None:
        embedding = OpenAiEmbedding()
        vector = embedding.embed_query("Hello world!")
        print(vector)

    def test_embed_document(self) -> None:
        embedding = OpenAiEmbedding()
        document = Document("你好，世界！", id="001")
        point = embedding.embed_document(document)
        print(point)

    def test_embed_documents(self) -> None:
        embedding = OpenAiEmbedding()
        doc1 = Document("你好，世界！", id="001")
        doc2 = Document("世界，你好！", id="002")
        points = embedding.embed_documents([doc1, doc2])
        print(points)


if __name__ == '__main__':
    unittest.main()
