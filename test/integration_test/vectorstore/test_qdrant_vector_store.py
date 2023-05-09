# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import unittest

import parameterized
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

from llmsdk.vectorstore import QdrantVectorStore
from llmsdk.embedding import MockEmbedding
from llmsdk.common import Document


class TestQdrantVectorStore(unittest.TestCase):

    def test_add_search(self):
        texts = ["foo", "bar", "baz"]
        embedding = MockEmbedding()
        points = embedding.embed_documents([Document(t) for t in texts])

        client = QdrantClient(location=":memory:")
        collection_name = "test"
        vectors_config = VectorParams(size=1024, distance=Distance.COSINE)
        client.create_collection(collection_name=collection_name,
                                 vectors_config=vectors_config)
        store = QdrantVectorStore(client=client,
                                  collection_name=collection_name)
        store.add_all(points)
        p = embedding.embed_query("foo")
        output = store.search(p.vector, limit=1)
        client.delete_collection(collection_name)
        store.close()
        self.assertEqual(output, p)


if __name__ == '__main__':
    unittest.main()
