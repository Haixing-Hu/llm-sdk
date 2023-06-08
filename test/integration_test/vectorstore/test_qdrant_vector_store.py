# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import copy
import unittest
import logging
from typing import Any

from llmsdk.vectorstore import (
    QdrantVectorStore,
    PayloadSchema,
    CollectionInfo,
)
from llmsdk.embedding import MockEmbedding, OpenAiEmbedding
from llmsdk.common import Document, DataType, Metadata, Distance
from llmsdk.criterion import equal

COLLECTION_NAME: str = "test"


class TestQdrantVectorStore(unittest.TestCase):

    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)

    def test_search(self):
        self._test_search(in_memory=True)
        self._test_search(host="127.0.0.1")
        self._test_search(path="/tmp/test_qdrant")

    def _test_search(self, **kwargs: Any):
        texts = ["foo", "bar", "baz"]
        documents = [Document(content=t, metadata=Metadata({"page": i}))
                     for i, t in enumerate(texts)]
        embedding = MockEmbedding()
        points = embedding.embed_documents(documents)
        store = QdrantVectorStore()
        store.open(**kwargs)
        try:
            store.create_collection(collection_name=COLLECTION_NAME,
                                    vector_size=embedding.output_dimensions)
            store.open_collection(COLLECTION_NAME)
            store.add_all(points)
            query = embedding.embed_query("foo")
            output = store.search(query, limit=1)
            output = [p.round_vector(MockEmbedding.PRECISION) for p in output]
            self.assertEqual(1, len(output))
            actual = output[0]
            self.assertEqual(query, actual.vector)
        finally:
            store.close_collection()
            store.delete_collection(COLLECTION_NAME)
            store.close()

    def test_search_with_filter(self):
        self._test_search_with_filter(in_memory=True)
        self._test_search_with_filter(path="/tmp/test_qdrant")
        self._test_search_with_filter(host="127.0.0.1")

    def _test_search_with_filter(self, **kwargs: Any):
        texts = ["foo", "bar", "baz"]
        documents = [Document(content=t, metadata=Metadata({"page": i}))
                     for i, t in enumerate(texts)]
        embedding = MockEmbedding()
        points = embedding.embed_documents(documents)
        store = QdrantVectorStore()
        store.open(**kwargs)
        try:
            store.create_collection(collection_name=COLLECTION_NAME,
                                    vector_size=embedding.output_dimensions)
            store.open_collection(COLLECTION_NAME)
            store.add_all(points)
            query = embedding.embed_query("foo")
            criterion = equal("page", 1)
            output = store.search(query, limit=1, criterion=criterion)
            output = [p.round_vector(MockEmbedding.PRECISION) for p in output]
            self.assertEqual(1, len(output))
            actual = output[0]
            expected = points[1]
            expected.metadata["page"] = 1
            expected.id = actual.id
            expected.score = actual.score
            self.assertEqual(expected, actual)
        finally:
            store.close_collection()
            store.delete_collection(COLLECTION_NAME)
            store.close()

    def test_max_marginal_relevance_search(self):
        self._test_max_marginal_relevance_search(in_memory=True)
        self._test_max_marginal_relevance_search(path="/tmp/test_qdrant")
        self._test_max_marginal_relevance_search(host="127.0.0.1")

    def _test_max_marginal_relevance_search(self, **kwargs: Any):
        texts = ["foo", "bar", "baz"]
        documents = [Document(content=t, metadata=Metadata({"page": i}))
                     for i, t in enumerate(texts)]
        embedding = MockEmbedding()
        points = embedding.embed_documents(documents)
        store = QdrantVectorStore()
        store.open(**kwargs)
        try:
            store.create_collection(collection_name=COLLECTION_NAME,
                                    vector_size=embedding.output_dimensions)
            store.open_collection(COLLECTION_NAME)
            store.add_all(points)
            query = embedding.embed_query("foo")
            output = store.max_marginal_relevance_search(query,
                                                         limit=2,
                                                         fetch_limit=3)
            output = [p.round_vector(MockEmbedding.PRECISION) for p in output]
            self.assertEqual(2, len(output))
            expected = [copy.deepcopy(points[0]),
                        copy.deepcopy(points[1])]
            expected[0].score = output[0].score
            expected[1].score = output[1].score
            self.assertEqual(expected, output)
        finally:
            store.close_collection()
            store.delete_collection(COLLECTION_NAME)
            store.close()

    def test_create_collection(self):
        self._test_create_collection(host="127.0.0.1")

    def _test_create_collection(self, **kwargs: Any):
        store = QdrantVectorStore()
        store.open(**kwargs)
        try:
            payload_schemas = [
                PayloadSchema(name="f1", type=DataType.INT),
                PayloadSchema(name="f2", type=DataType.FLOAT),
                PayloadSchema(name="f3", type=DataType.STRING),
                PayloadSchema(name="f4", type=DataType.STRING),
            ]
            store.create_collection(collection_name=COLLECTION_NAME,
                                    vector_size=10,
                                    payload_schemas=payload_schemas)
            info = store.get_collection_info(COLLECTION_NAME)
            info.payload_schemas.sort()
            print(info)
            expected = CollectionInfo(name=COLLECTION_NAME,
                                      size=0,
                                      vector_dimension=10,
                                      distance=Distance.COSINE,
                                      payload_schemas=payload_schemas)
            self.assertEqual(expected, info)
        finally:
            store.delete_collection(COLLECTION_NAME)
            store.close()

    def test_has_collection__non_exist_collection_localhost(self):
        self._test_has_collection(in_memory=True)
        self._test_has_collection(host="127.0.0.1")
        self._test_has_collection(path="/tmp/test_qdrant")

    def _test_has_collection(self, **kwargs: Any):
        store = QdrantVectorStore()
        store.open(**kwargs)
        try:
            result = store.has_collection(COLLECTION_NAME)
            self.assertEqual(False, result)
            store.create_collection(collection_name=COLLECTION_NAME,
                                    vector_size=10)
            result = store.has_collection(COLLECTION_NAME)
            self.assertEqual(True, result)
        finally:
            store.delete_collection(COLLECTION_NAME)
            store.close()

    def test_collection_size(self):
        self._test_collection_size(in_memory=True)
        self._test_collection_size(host="127.0.0.1")
        self._test_collection_size(path="/tmp/test_qdrant")

    def _test_collection_size(self, **kwargs: Any):
        texts = ["foo", "bar", "baz"]
        documents = [Document(content=t, metadata=Metadata({"page": i}))
                     for i, t in enumerate(texts)]
        embedding = MockEmbedding()
        points = embedding.embed_documents(documents)
        store = QdrantVectorStore()
        store.open(**kwargs)
        try:
            store.create_collection(collection_name=COLLECTION_NAME,
                                    vector_size=embedding.output_dimensions)
            result = store.has_collection(COLLECTION_NAME)
            self.assertEqual(True, result)
            store.open_collection(COLLECTION_NAME)
            store.add(points[0])
            info = store.get_collection_info(COLLECTION_NAME)
            self.assertEqual(1, info.size)
            store.add(points[1])
            info = store.get_collection_info(COLLECTION_NAME)
            self.assertEqual(2, info.size)
            store.add(points[2])
            info = store.get_collection_info(COLLECTION_NAME)
            self.assertEqual(3, info.size)
        finally:
            store.close_collection()
            store.delete_collection(COLLECTION_NAME)
            store.close()

    def test_similarity_search_with_score_threshold(self):
        self._test_similarity_search_with_score_threshold(in_memory=True)

    def _test_similarity_search_with_score_threshold(self, **kwargs: Any):
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
        documents = [Document(content=q) for q in questions]
        embedding = OpenAiEmbedding()
        points = embedding.embed_documents(documents)
        vector = embedding.embed_text("南京宁慧保是什么")
        store = QdrantVectorStore()
        store.open(**kwargs)
        try:
            store.create_collection(collection_name=COLLECTION_NAME,
                                    vector_size=embedding.output_dimensions)
            store.open_collection(COLLECTION_NAME)
            store.add_all(points)
            result_points = store.similarity_search(vector,
                                                    limit=len(questions),
                                                    score_threshold=0.8)
            result = Document.from_points(result_points)
            print(result)
        finally:
            store.close_collection()
            store.delete_collection(COLLECTION_NAME)
            store.close()


if __name__ == '__main__':
    unittest.main()
