# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
import unittest
import logging
from typing import List, Optional

from llmsdk.embedding import OpenAiEmbedding
from llmsdk.common import Document, Distance
# from llmsdk.util.openai_utils import set_openai_debug_mode


class TestOpenAiEmbedding(unittest.TestCase):
    def assertListAlmostEqual(self,
                              list1: List[float],
                              list2: List[float],
                              places: int = 3,
                              delta: Optional[float] = None,
                              msg: Optional[str] = None):
        self.assertEqual(len(list1), len(list2))
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(a, b, places=places, delta=delta, msg=msg)

    def setUp(self) -> None:
        logging.basicConfig(level=logging.INFO)
        # set_openai_debug_mode()

    def test_embed_query(self) -> None:
        embedding = OpenAiEmbedding()
        vector = embedding.embed_query("Hello, world!")
        print(vector)

    def test_embed_document(self) -> None:
        embedding = OpenAiEmbedding()
        document = Document("Hello, world!", id="001")
        point = embedding.embed_document(document)
        print(point)

    def test_embed_documents(self) -> None:
        embedding = OpenAiEmbedding()
        doc1 = Document("Hello, world!", id="001")
        doc2 = Document("World, hello!", id="002")
        points = embedding.embed_documents([doc1, doc2])
        print(points)

    def test_embed_consistent(self) -> None:
        embedding = OpenAiEmbedding()
        questions = [
            "什么是“南京宁慧保”？",
            "这款产品是哪个保险公司承保的？",
            "城市型补充医疗是什么？",
            "这款产品保障范围是什么？",
            "什么是门诊大病？",
            "南京门诊大病指哪些病？",
            "住院医疗费用有哪些？",
            "什么叫必需且合理的医疗费用？",
            "中草药费用报销吗？",
            "门诊特定项目是什么？",
            "这款产品的赔付比例是多少？",
        ]
        print("Embedding vectors 1")
        vectors1 = embedding.embed_texts(questions)
        query = embedding.embed_text("南京宁慧保是什么")
        print(f"query={query}")
        scores = [Distance.COSINE.between(query, v) for v in vectors1]
        print(scores)

    def test_cache(self):
        embedding = OpenAiEmbedding(use_cache=True, cache_size=3)
        doc1 = Document("Hello, world!", id="001")
        doc2 = Document("World, hello!", id="002")
        doc3 = Document("What's your name?", id="003")
        doc4 = Document("World, hello!", id="004")
        doc5 = Document("What's your name?", id="005")
        points = embedding.embed_documents([doc1, doc2, doc3, doc4, doc5])
        print(f"Embedded {len(points)} points.")
        self.assertIsNotNone(embedding.cache)
        print(f"Embedding cache keys: {list(embedding.cache.keys())}")

        doc6 = Document("World, hello!", id="006")
        doc7 = Document("What's your name?", id="007")
        doc8 = Document("World, hello!", id="008")
        doc9 = Document("Here you are!", id="009")
        doc10 = Document("Hello, world!", id="010")
        points = embedding.embed_documents([doc6, doc7, doc8, doc9, doc10])
        print(f"Embedded {len(points)} points.")
        self.assertIsNotNone(embedding.cache)
        print(f"Embedding cache keys: {list(embedding.cache.keys())}")


if __name__ == '__main__':
    unittest.main()
