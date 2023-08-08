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

from llmsdk.embedding import OpenAiEmbedding
from llmsdk.vectorstore import QdrantVectorStore
from llmsdk.splitter import CharacterTextSplitter
from llmsdk.llm import ChatGpt
from llmsdk.retriever import SimilarRecordRetriever


class TestSimilarRecordRetriever(unittest.TestCase):

    TEST_DATA = [{
        "编码": "0001",
        "名称": "一次性使用球囊扩张导管",
        "药品规格": "",
        "包装规格": "球囊直径4.0mm*球囊长度20mm",
        "药品商品名": "",
        "中文通用名": "",
    }, {
        "编码": "0002",
        "名称": "一次性使用球囊扩张导管",
        "药品规格": "",
        "包装规格": "球囊直径2.75mm*球囊长度25mm",
        "药品商品名": "",
        "中文通用名": "",
    }, {
        "编码": "0003",
        "名称": "髋关节假体金属髋臼(金属髋臼CL-II型)",
        "药品规格": "38#",
        "包装规格": "",
        "药品商品名": "髋关节假体金属髋臼(金属髋臼CL - II型)",
        "中文通用名": "107 - 人工关节",
    }, {
        "编码": "0004",
        "名称": "一次性使用负压采血管",
        "药品规格": "促凝管+分离胶 1ml",
        "包装规格": "",
        "药品商品名": "一次性使用负压采血管",
        "中文通用名": "203-采血管",
    }, {
        "编码": "0005",
        "名称": "一次性使用负压采血管",
        "药品规格": "EDTAK2 9ml",
        "包装规格": "",
        "药品商品名": "一次性使用负压采血管",
        "中文通用名": "203-采血管",
    }, {
        "编码": "0006",
        "名称": "大山楂颗粒 ",
        "药品规格": "每袋装10g(低糖型)",
        "包装规格": "每袋装15g:每袋装10g(低糖型)",
        "药品商品名": "",
        "中文通用名": "统筹外中成药",
    }, {
        "编码": "0007",
        "名称": "大山楂丸",
        "药品规格": "每丸重9g",
        "包装规格": "每丸重9g",
        "药品商品名": "",
        "中文通用名": "统筹外中成药",
    }, {
        "编码": "0008",
        "名称": "九制大黄丸",
        "药品规格": "每50粒重3g,6g",
        "包装规格": "每50粒重3g",
        "药品商品名": "",
        "中文通用名": "统筹外中成药",
    }]

    retriever = None

    @classmethod
    def setUpClass(cls):
        logging.basicConfig(level=logging.INFO)
        store = QdrantVectorStore()
        embedding = OpenAiEmbedding()
        splitter = CharacterTextSplitter()
        collection_name = "DRUG_MATCH"
        llm = ChatGpt(temperature=1)
        cls.retriever = SimilarRecordRetriever(record_id_field="编码",
                                               vector_store=store,
                                               collection_name=collection_name,
                                               embedding=embedding,
                                               splitter=splitter,
                                               llm=llm,
                                               language="zh_CN")
        cls.retriever.open(in_memory=True)
        try:
            for r in cls.TEST_DATA:
                cls.retriever.add_record(r)
        except Exception as e:
            cls.retriever.close()
            raise e

    @classmethod
    def tearDownClass(cls):
        cls.retriever.close()

    def test_find(self):
        # self.retriever.set_logging_level(logging.DEBUG)
        f1 = self.retriever.find({"名称": "山楂颗粒"})
        self.assertEqual(self.TEST_DATA[5], f1)
        f3 = self.retriever.find({"名称": "采血管", "规格": "1ml"})
        self.assertEqual(self.TEST_DATA[3], f3)
        f5 = self.retriever.find({"名称": "采血管9ml"})
        self.assertEqual(self.TEST_DATA[4], f5)
        f8 = self.retriever.find({"名称": "大黄丸"})
        self.assertEqual(self.TEST_DATA[7], f8)

    def test_find__non_name_field_match(self):
        f2 = self.retriever.find({"名称": "人工关节"})
        self.assertEqual(self.TEST_DATA[2], f2)

    def test_find__no_match_1(self):
        f4 = self.retriever.find({"名称": "采血管12ml"})
        self.assertIsNone(f4)

    def test_find__more_than_one_match(self):
        f6 = self.retriever.find({"名称": "中成药"})
        self.assertIsNone(f6)


if __name__ == '__main__':
    unittest.main()
