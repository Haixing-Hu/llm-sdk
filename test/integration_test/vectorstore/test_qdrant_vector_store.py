# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import unittest
import logging

from qdrant_client import QdrantClient

from llmsdk.vectorstore import QdrantVectorStore, PayloadSchema, DataType, CollectionInfo, Distance
from llmsdk.embedding import MockEmbedding
from llmsdk.common import Document
from llmsdk.criterion import equal


class TestQdrantVectorStore(unittest.TestCase):

    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)

    def test_search(self):
        collection_name = "test"
        vector_size = MockEmbedding.VECTOR_DIMENSION
        texts = ["foo", "bar", "baz"]
        documents = [Document(t, metadata={"page": i}) for i, t in enumerate(texts)]
        embedding = MockEmbedding()
        points = embedding.embed_documents(documents)
        client = QdrantClient(location=":memory:")
        store = QdrantVectorStore(client)
        store.open()
        try:
            store.create_collection(collection_name=collection_name,
                                    vector_size=vector_size)
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
        vector_size = MockEmbedding.VECTOR_DIMENSION
        texts = ["foo", "bar", "baz"]
        documents = [Document(t, metadata={"page": i}) for i, t in enumerate(texts)]
        embedding = MockEmbedding()
        points = embedding.embed_documents(documents)
        client = QdrantClient(location=":memory:")
        store = QdrantVectorStore(client)
        store.open()
        try:
            store.create_collection(collection_name=collection_name,
                                    vector_size=vector_size)
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

    def test_create_collection(self):
        collection_name = "test"
        vector_size = MockEmbedding.VECTOR_DIMENSION
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
                                    vector_size=vector_size,
                                    payload_schemas=payload_schemas)
            info = store.get_collection_info(collection_name)
            info.payload_schemas.sort()
            print(info)
            expected = CollectionInfo(name=collection_name,
                                      size=0,
                                      vector_size=vector_size,
                                      distance=Distance.COSINE,
                                      payload_schemas=payload_schemas)
            self.assertEqual(expected, info)
        finally:
            store.delete_collection(collection_name)
            store.close()


if __name__ == '__main__':
    unittest.main()
