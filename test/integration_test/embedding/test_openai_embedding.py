# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import unittest
import logging
from typing import List, Optional

from llmsdk.embedding import OpenAiEmbedding
from llmsdk.common import Document, Distance
from llmsdk.util.openai_utils import set_openai_debug_mode


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
        logging.basicConfig(level=logging.DEBUG)
        set_openai_debug_mode()

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

    def test_embed_consistent(self) -> None:
        embedding = OpenAiEmbedding()
        # v1 = embedding.embed_text("你好，世界！")
        # v2 = embedding.embed_text("你好，世界！")
        # self.assertListAlmostEqual(v1, v2)
        # vectors = embedding.embed_texts(["你好，世界！", "你好，世界！"])
        # self.assertListAlmostEqual(vectors[0], vectors[1])

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
        # print(f"vectors1 = {vectors1}")
        # print("Embedding vectors 2")
        # vectors2 = embedding.embed_texts(questions)
        # print(f"vectors2 = {vectors2}")
        # for i, _ in enumerate(vectors1):
        #     self.assertListAlmostEqual(vectors1[i], vectors2[i],
        #                                msg=f"i={i}, text={questions[i]}")
        query = embedding.embed_text("南京宁慧保是什么")
        print(f"query={query}")
        scores = [Distance.COSINE.between(query, v) for v in vectors1]
        print(scores)


if __name__ == '__main__':
    unittest.main()
