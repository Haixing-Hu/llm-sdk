# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
import unittest

import numpy as np

from llmsdk.vectorstore.vector_store_utils import maximal_marginal_relevance


class TestVectorStoreUtils(unittest.TestCase):

    def test_maximal_marginal_relevance_lambda_zero(self):
        query_embedding = np.random.random(size=5)
        embedding_list = [query_embedding, query_embedding, np.zeros(5)]
        expected = [0, 2]
        actual = maximal_marginal_relevance(
            query_vector=query_embedding,
            similarity_vectors=embedding_list,
            lambda_multipy=0,
            limit=2,
        )
        self.assertEqual(expected, actual)

    def test_maximal_marginal_relevance_lambda_one(self):
        query_embedding = np.random.random(size=5)
        embedding_list = [query_embedding, query_embedding, np.zeros(5)]
        expected = [0, 1]
        actual = maximal_marginal_relevance(
            query_vector=query_embedding,
            similarity_vectors=embedding_list,
            lambda_multipy=1,
            limit=2,
        )
        self.assertEqual(expected, actual)

    def test_maximal_marginal_relevance(self):
        query_embedding = np.array([1, 0])
        # Vectors that are 30, 45 and 75 degrees from query vector (cosine similarity of
        # 0.87, 0.71, 0.26) and the latter two are 15 and 60 degree from the first
        # (cosine similarity 0.97 and 0.71). So for 3rd vector be chosen, must be case that
        # 0.71lambda - 0.97(1 - lambda) < 0.26lambda - 0.71(1-lambda)
        # -> lambda ~< .26 / .71
        embedding_list = [[3**0.5, 1], [1, 1], [1, 2 + (3**0.5)]]
        expected = [0, 2]
        actual = maximal_marginal_relevance(
            query_vector=query_embedding,
            similarity_vectors=embedding_list,
            lambda_multipy=(25 / 71),
            limit=2,
        )
        self.assertEqual(expected, actual)

        expected = [0, 1]
        actual = maximal_marginal_relevance(
            query_vector=query_embedding,
            similarity_vectors=embedding_list,
            lambda_multipy=(27 / 71),
            limit=2,
        )
        self.assertEqual(expected, actual)

    def test_maximal_marginal_relevance_query_dim(self):
        query_embedding = np.random.random(size=5)
        query_embedding_2d = query_embedding.reshape((1, 5))
        embedding_list = np.random.random(size=(4, 5)).tolist()
        first = maximal_marginal_relevance(
            query_embedding,
            embedding_list
        )
        second = maximal_marginal_relevance(
            query_embedding_2d,
            embedding_list
        )
        self.assertEqual(first, second)


if __name__ == '__main__':
    unittest.main()
