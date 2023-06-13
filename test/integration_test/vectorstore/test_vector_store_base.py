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
    VectorStore,
    PayloadSchema,
    CollectionInfo,
)
from llmsdk.embedding import MockEmbedding, OpenAiEmbedding
from llmsdk.common import Document, DataType, Metadata, Distance
from llmsdk.criterion import equal

COLLECTION_NAME: str = "test"


class TestVectorStoreBase(unittest.TestCase):
    """
    The base class for integration testing the vector stores.
    """

    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)

    def _test_search(self, store: VectorStore, **kwargs: Any):
        texts = ["foo", "bar", "baz"]
        documents = [Document(content=t, metadata=Metadata({"page": i}))
                     for i, t in enumerate(texts)]
        embedding = MockEmbedding()
        points = embedding.embed_documents(documents)
        store.open(**kwargs)
        try:
            store.create_collection(collection_name=COLLECTION_NAME,
                                    vector_size=embedding.vector_dimension)
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

    def _test_search_with_filter(self, store: VectorStore, **kwargs: Any):
        texts = ["foo", "bar", "baz"]
        documents = [Document(content=t, metadata=Metadata({"page": i}))
                     for i, t in enumerate(texts)]
        embedding = MockEmbedding()
        points = embedding.embed_documents(documents)
        store.open(**kwargs)
        try:
            store.create_collection(collection_name=COLLECTION_NAME,
                                    vector_size=embedding.vector_dimension)
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

    def _test_mmr_search(self, store: VectorStore, **kwargs: Any):
        texts = ["foo", "bar", "baz"]
        documents = [Document(content=t, metadata=Metadata({"page": i}))
                     for i, t in enumerate(texts)]
        embedding = MockEmbedding()
        points = embedding.embed_documents(documents)
        store.open(**kwargs)
        try:
            store.create_collection(collection_name=COLLECTION_NAME,
                                    vector_size=embedding.vector_dimension)
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

    def _test_create_collection(self, store: VectorStore, **kwargs: Any):
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

    def _test_has_collection(self, store: VectorStore, **kwargs: Any):
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

    def _test_collection_size(self, store: VectorStore, **kwargs: Any):
        texts = ["foo", "bar", "baz"]
        documents = [Document(content=t, metadata=Metadata({"page": i}))
                     for i, t in enumerate(texts)]
        embedding = MockEmbedding()
        points = embedding.embed_documents(documents)
        store.open(**kwargs)
        try:
            store.create_collection(collection_name=COLLECTION_NAME,
                                    vector_size=embedding.vector_dimension)
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

    def _test_similarity_search(self, store: VectorStore, **kwargs: Any):
        questions = [
            "什么是“惠民保”？",
            "这款产品是哪个保险公司承保的？",
            "这款产品保障范围是什么？",
            "什么是门诊大病？",
            "南京门诊大病指哪些病？",
            "住院医疗费用有哪些？",
        ]
        documents = [Document(content=q) for q in questions]
        embedding = OpenAiEmbedding()
        points = embedding.embed_documents(documents)
        vector = embedding.embed_text("惠民保是什么")
        store.open(**kwargs)
        try:
            store.create_collection(collection_name=COLLECTION_NAME,
                                    vector_size=embedding.vector_dimension)
            store.open_collection(COLLECTION_NAME)
            store.add_all(points)
            result_points = store.similarity_search(vector,
                                                    limit=len(questions),
                                                    score_threshold=0.9)
            result = Document.from_points(result_points)
            print(result)
            self.assertEqual(1, len(result))
            self.assertEqual("什么是“惠民保”？", result[0].content)
        finally:
            store.close_collection()
            store.delete_collection(COLLECTION_NAME)
            store.close()


if __name__ == '__main__':
    unittest.main()
