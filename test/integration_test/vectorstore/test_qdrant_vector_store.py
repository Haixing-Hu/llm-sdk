# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import unittest
import logging

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, Filter, FieldCondition, MatchValue

from llmsdk.vectorstore import VectorStore, QdrantVectorStore
from llmsdk.embedding import MockEmbedding
from llmsdk.common import Document
from llmsdk.criterion import equal


def prepare_store() -> VectorStore:
    client = QdrantClient(location=":memory:")
    collection_name = "test"
    vectors_config = VectorParams(size=MockEmbedding.VECTOR_DIMENSION,
                                  distance=Distance.COSINE)
    client.create_collection(collection_name=collection_name,
                             vectors_config=vectors_config)

    store = QdrantVectorStore(client=client)
    store.open()
    store.open_collection(collection_name)
    return store


class TestQdrantVectorStore(unittest.TestCase):

    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)

    def test_search(self):
        texts = ["foo", "bar", "baz"]
        documents = [Document(t, {"page": i}) for i, t in enumerate(texts)]
        embedding = MockEmbedding()
        points = embedding.embed_documents(documents)

        store = prepare_store()
        try:
            store.add_all(points)
            query = embedding.embed_query("foo")
            output = store.search(query.vector, limit=1)
            self.assertEqual(1, len(output))
            actual = output[0]

            query.metadata["page"] = 0
            query.id = actual.id
            query.score = actual.score
            self.assertEqual(query, actual)
        finally:
            store.close()

    def test_search_with_filter(self):
        texts = ["foo", "bar", "baz"]
        documents = [Document(t, {"page": i}) for i, t in enumerate(texts)]
        embedding = MockEmbedding()
        points = embedding.embed_documents(documents)

        store = prepare_store()
        try:
            store.add_all(points)
            query = embedding.embed_query("foo")
            criterion = equal("page", 1)
            output = store.search(query.vector, limit=1, criterion=criterion)
            self.assertEqual(1, len(output))
            actual = output[0]

            expected = points[1]
            expected.metadata["page"] = 1
            expected.id = actual.id
            expected.score = actual.score
            self.assertEqual(expected, actual)
        finally:
            store.close()


if __name__ == '__main__':
    unittest.main()
