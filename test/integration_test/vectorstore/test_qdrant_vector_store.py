# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import unittest
from llmsdk.vectorstore import QdrantVectorStore

from .test_vector_store_base import TestVectorStoreBase


class TestQdrantVectorStore(TestVectorStoreBase):

    def test_search(self):
        self._test_search(store=QdrantVectorStore(), in_memory=True)
        self._test_search(store=QdrantVectorStore(), host="127.0.0.1")
        self._test_search(store=QdrantVectorStore(), path="/tmp/test_qdrant")

    def test_search_with_filter(self):
        self._test_search_with_filter(store=QdrantVectorStore(), in_memory=True)
        self._test_search_with_filter(store=QdrantVectorStore(), path="/tmp/test_qdrant")
        self._test_search_with_filter(store=QdrantVectorStore(), host="127.0.0.1")

    def test_mmr_search(self):
        self._test_mmr_search(store=QdrantVectorStore(), in_memory=True)
        self._test_mmr_search(store=QdrantVectorStore(), path="/tmp/test_qdrant")
        self._test_mmr_search(store=QdrantVectorStore(), host="127.0.0.1")

    def test_create_collection(self):
        self._test_create_collection(store=QdrantVectorStore(), host="127.0.0.1")

    def test_has_collection(self):
        self._test_has_collection(store=QdrantVectorStore(), in_memory=True)
        self._test_has_collection(store=QdrantVectorStore(), host="127.0.0.1")
        self._test_has_collection(store=QdrantVectorStore(), path="/tmp/test_qdrant")

    def test_collection_size(self):
        self._test_collection_size(store=QdrantVectorStore(), in_memory=True)
        self._test_collection_size(store=QdrantVectorStore(), host="127.0.0.1")
        self._test_collection_size(store=QdrantVectorStore(), path="/tmp/test_qdrant")

    def test_similarity_search_with_score_threshold(self):
        self._test_similarity_search(store=QdrantVectorStore(), in_memory=True)


if __name__ == '__main__':
    unittest.main()
