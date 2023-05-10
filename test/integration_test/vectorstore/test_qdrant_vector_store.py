# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import unittest
import logging

import parameterized
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

from llmsdk.vectorstore import QdrantVectorStore
from llmsdk.embedding import MockEmbedding
from llmsdk.common import Document


class TestQdrantVectorStore(unittest.TestCase):

    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)

    def test_add_search(self):
        texts = ["foo", "bar", "baz"]
        embedding = MockEmbedding()
        points = embedding.embed_documents([Document(t) for t in texts])

        client = QdrantClient(location=":memory:")
        collection_name = "test"
        vector_size = len(points[0].vector)
        vectors_config = VectorParams(size=vector_size, distance=Distance.COSINE)
        client.create_collection(collection_name=collection_name,
                                 vectors_config=vectors_config)
        store = QdrantVectorStore(client=client,
                                  collection_name=collection_name)
        store.add_all(points)
        expected = embedding.embed_query("foo")
        output = store.search(expected.vector, limit=1)
        client.delete_collection(collection_name)
        store.close()
        self.assertEqual(1, len(output))
        actual = output[0]
        expected.id = actual.id
        expected.score = actual.score
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
