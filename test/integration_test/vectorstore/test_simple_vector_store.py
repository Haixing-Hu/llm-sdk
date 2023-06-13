# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import unittest

from llmsdk.vectorstore import SimpleVectorStore

from .test_vector_store_base import TestVectorStoreBase


class TestSimpleVectorStore(TestVectorStoreBase):

    def test_search(self):
        self._test_search(store=SimpleVectorStore())

    def test_search_with_filter(self):
        self._test_search_with_filter(store=SimpleVectorStore())

    def test_mmr_search(self):
        self._test_mmr_search(store=SimpleVectorStore())

    def test_create_collection(self):
        self._test_create_collection(store=SimpleVectorStore())

    def test_has_collection(self):
        self._test_has_collection(store=SimpleVectorStore())

    def test_collection_size(self):
        self._test_collection_size(store=SimpleVectorStore())

    def test_similarity_search_with_score_threshold(self):
        self._test_similarity_search(store=SimpleVectorStore())


if __name__ == '__main__':
    unittest.main()
