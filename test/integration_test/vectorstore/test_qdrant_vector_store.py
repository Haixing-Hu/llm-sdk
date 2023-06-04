# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import copy
import unittest
import logging

from llmsdk.vectorstore import (
    QdrantVectorStore,
    PayloadSchema,
    CollectionInfo,
    Distance,
)
from llmsdk.embedding import MockEmbedding
from llmsdk.common import Document, DataType, Metadata
from llmsdk.criterion import equal

COLLECTION_NAME: str = "test"


class TestQdrantVectorStore(unittest.TestCase):

    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)

    def test_search(self):
        texts = ["foo", "bar", "baz"]
        documents = [Document(content=t, metadata=Metadata({"page": i}))
                     for i, t in enumerate(texts)]
        embedding = MockEmbedding()
        points = embedding.embed_documents(documents)
        store = QdrantVectorStore(in_memory=True)
        store.open()
        try:
            store.create_collection(collection_name=COLLECTION_NAME,
                                    vector_size=embedding.output_dimensions)
            store.open_collection(COLLECTION_NAME)
            store.add_all(points)
            query = embedding.embed_query("foo")
            output = store.search(query, limit=1)
            self.assertEqual(1, len(output))
            actual = output[0]
            self.assertEqual(query, actual.vector)
        finally:
            store.delete_collection(COLLECTION_NAME)
            store.close()

    def test_search_with_filter(self):
        texts = ["foo", "bar", "baz"]
        documents = [Document(content=t, metadata=Metadata({"page": i}))
                     for i, t in enumerate(texts)]
        embedding = MockEmbedding()
        points = embedding.embed_documents(documents)
        store = QdrantVectorStore(in_memory=True)
        store.open()
        try:
            store.create_collection(collection_name=COLLECTION_NAME,
                                    vector_size=embedding.output_dimensions)
            store.open_collection(COLLECTION_NAME)
            store.add_all(points)
            query = embedding.embed_query("foo")
            criterion = equal("page", 1)
            output = store.search(query, limit=1, criterion=criterion)
            self.assertEqual(1, len(output))
            actual = output[0]
            expected = points[1]
            expected.metadata["page"] = 1
            expected.id = actual.id
            expected.score = actual.score
            self.assertEqual(expected, actual)
        finally:
            store.delete_collection(COLLECTION_NAME)
            store.close()

    def test_max_marginal_relevance_search(self):
        texts = ["foo", "bar", "baz"]
        documents = [Document(content=t, metadata=Metadata({"page": i}))
                     for i, t in enumerate(texts)]
        embedding = MockEmbedding()
        points = embedding.embed_documents(documents)
        store = QdrantVectorStore(in_memory=True)
        store.open()
        try:
            store.create_collection(collection_name=COLLECTION_NAME,
                                    vector_size=embedding.output_dimensions)
            store.open_collection(COLLECTION_NAME)
            store.add_all(points)
            query = embedding.embed_query("foo")
            output = store.max_marginal_relevance_search(query,
                                                         limit=2,
                                                         fetch_limit=3)
            self.assertEqual(2, len(output))
            expected = [copy.deepcopy(points[0]),
                        copy.deepcopy(points[1])]
            expected[0].score = output[0].score
            expected[1].score = output[1].score
            self.assertEqual(expected, output)
        finally:
            store.delete_collection(COLLECTION_NAME)
            store.close()

    def test_create_collection(self):
        store = QdrantVectorStore(host="127.0.0.1")
        store.open()
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
                                      vector_size=10,
                                      distance=Distance.COSINE,
                                      payload_schemas=payload_schemas)
            self.assertEqual(expected, info)
        finally:
            store.delete_collection(COLLECTION_NAME)
            store.close()

    def test_has_collection__non_exist_collection_localhost(self):
        store = QdrantVectorStore(host="127.0.0.1")
        store.open()
        try:
            result = store.has_collection(COLLECTION_NAME)
            self.assertEqual(False, result)
        finally:
            store.close()

    def test_has_collection__non_exist_collection_memory(self):
        store = QdrantVectorStore(in_memory=True)
        store.open()
        try:
            result = store.has_collection(COLLECTION_NAME)
            self.assertEqual(False, result)
        finally:
            store.close()

    def test_has_collection__exist_collection_localhost(self):
        store = QdrantVectorStore(host="127.0.0.1")
        store.open()
        try:
            store.create_collection(collection_name=COLLECTION_NAME,
                                    vector_size=10)
            result = store.has_collection(COLLECTION_NAME)
            self.assertEqual(True, result)
        finally:
            store.delete_collection(COLLECTION_NAME)
            store.close()

    def test_has_collection__exist_collection_memory(self):
        store = QdrantVectorStore(in_memory=True)
        store.open()
        try:
            store.create_collection(collection_name=COLLECTION_NAME,
                                    vector_size=10)
            result = store.has_collection(COLLECTION_NAME)
            self.assertEqual(True, result)
        finally:
            store.delete_collection(COLLECTION_NAME)
            store.close()

    def test_collection_size__in_memory(self):
        store = QdrantVectorStore(in_memory=True)
        self._test_collection_size(store)

    def test_collection_size__localhost(self):
        store = QdrantVectorStore(host="127.0.0.1")
        self._test_collection_size(store)

    def test_collection_size__file(self):
        store = QdrantVectorStore(path="/tmp/test_qdrant")
        self._test_collection_size(store)

    def _test_collection_size(self, store: QdrantVectorStore):
        texts = ["foo", "bar", "baz"]
        documents = [Document(content=t, metadata=Metadata({"page": i}))
                     for i, t in enumerate(texts)]
        embedding = MockEmbedding()
        points = embedding.embed_documents(documents)
        store.open()
        try:
            store.create_collection(collection_name=COLLECTION_NAME,
                                    vector_size=10)
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
            store.delete_collection(COLLECTION_NAME)
            store.close()


if __name__ == '__main__':
    unittest.main()
