# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import copy
import unittest
import logging

from qdrant_client import QdrantClient

from llmsdk.vectorstore import (
    QdrantVectorStore,
    PayloadSchema,
    CollectionInfo,
    Distance,
)
from llmsdk.embedding import MockEmbedding
from llmsdk.common import Document, DataType, Metadata
from llmsdk.criterion import equal


class TestQdrantVectorStore(unittest.TestCase):

    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)

    def test_search(self):
        collection_name = "test"
        texts = ["foo", "bar", "baz"]
        documents = [Document(content=t, metadata=Metadata({"page": i}))
                     for i, t in enumerate(texts)]
        embedding = MockEmbedding()
        points = embedding.embed_documents(documents)
        client = QdrantClient(location=":memory:")
        store = QdrantVectorStore(client)
        store.open()
        try:
            store.create_collection(collection_name=collection_name,
                                    vector_size=embedding.output_dimensions)
            store.open_collection(collection_name)
            store.add_all(points)
            query = embedding.embed_query("foo")
            output = store.search(query, limit=1)
            self.assertEqual(1, len(output))
            actual = output[0]
            self.assertEqual(query, actual.vector)
        finally:
            store.delete_collection(collection_name)
            store.close()

    def test_search_with_filter(self):
        collection_name = "test"
        texts = ["foo", "bar", "baz"]
        documents = [Document(content=t, metadata=Metadata({"page": i}))
                     for i, t in enumerate(texts)]
        embedding = MockEmbedding()
        points = embedding.embed_documents(documents)
        client = QdrantClient(location=":memory:")
        store = QdrantVectorStore(client)
        store.open()
        try:
            store.create_collection(collection_name=collection_name,
                                    vector_size=embedding.output_dimensions)
            store.open_collection(collection_name)
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
            store.delete_collection(collection_name)
            store.close()

    def test_max_marginal_relevance_search(self):
        collection_name = "test"
        texts = ["foo", "bar", "baz"]
        documents = [Document(content=t, metadata=Metadata({"page": i}))
                     for i, t in enumerate(texts)]
        embedding = MockEmbedding()
        points = embedding.embed_documents(documents)
        client = QdrantClient(location=":memory:")
        store = QdrantVectorStore(client)
        store.open()
        try:
            store.create_collection(collection_name=collection_name,
                                    vector_size=embedding.output_dimensions)
            store.open_collection(collection_name)
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
            store.delete_collection(collection_name)
            store.close()

    def test_create_collection(self):
        collection_name = "test"
        client = QdrantClient(location="127.0.0.1")
        store = QdrantVectorStore(client)
        store.open()
        try:
            payload_schemas = [
                PayloadSchema(name="f1", type=DataType.INT),
                PayloadSchema(name="f2", type=DataType.FLOAT),
                PayloadSchema(name="f3", type=DataType.STRING),
                PayloadSchema(name="f4", type=DataType.STRING),
            ]
            store.create_collection(collection_name=collection_name,
                                    vector_size=10,
                                    payload_schemas=payload_schemas)
            info = store.get_collection_info(collection_name)
            info.payload_schemas.sort()
            print(info)
            expected = CollectionInfo(name=collection_name,
                                      size=0,
                                      vector_size=10,
                                      distance=Distance.COSINE,
                                      payload_schemas=payload_schemas)
            self.assertEqual(expected, info)
        finally:
            store.delete_collection(collection_name)
            store.close()

    def test_has_collection__non_exist_collection_localhost(self):
        collection_name = "test"
        client = QdrantClient(location="127.0.0.1")
        store = QdrantVectorStore(client)
        store.open()
        try:
            result = store.has_collection(collection_name)
            self.assertEqual(False, result)
        finally:
            store.close()

    def test_has_collection__non_exist_collection_memory(self):
        collection_name = "test"
        client = QdrantClient(location=":memory:")
        store = QdrantVectorStore(client)
        store.open()
        try:
            result = store.has_collection(collection_name)
            self.assertEqual(False, result)
        finally:
            store.close()

    def test_has_collection__exist_collection_localhost(self):
        collection_name = "test"
        client = QdrantClient(location="127.0.0.1")
        store = QdrantVectorStore(client)
        store.open()
        try:
            store.create_collection(collection_name=collection_name,
                                    vector_size=10)
            result = store.has_collection(collection_name)
            self.assertEqual(True, result)
        finally:
            store.delete_collection(collection_name)
            store.close()

    def test_has_collection__exist_collection_memory(self):
        collection_name = "test"
        client = QdrantClient(location=":memory:")
        store = QdrantVectorStore(client)
        store.open()
        try:
            store.create_collection(collection_name=collection_name,
                                    vector_size=10)
            result = store.has_collection(collection_name)
            self.assertEqual(True, result)
        finally:
            store.delete_collection(collection_name)
            store.close()


if __name__ == '__main__':
    unittest.main()
