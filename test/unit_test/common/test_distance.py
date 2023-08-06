# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
import unittest
from numpy.testing import assert_almost_equal

from llmsdk.common import Distance, Point


class DistanceTestCase(unittest.TestCase):
    def setUp(self):
        self.points = [
            Point(vector=[1, 2, 3], score=0.8),
            Point(vector=[4, 5, 6], score=0.5),
            Point(vector=[7, 8, 9], score=0.2)
        ]
        self.query_vector = [1, 1, 1]

    def test_euclid_distance_between(self):
        v1 = [1, 2, 3]
        v2 = [4, 5, 6]
        expected_distance = 5.196152
        calculated_distance = Distance.EUCLID.between(v1, v2)
        assert_almost_equal(calculated_distance, expected_distance, decimal=6)

    def test_cosine_distance_between(self):
        v1 = [1, 2, 3]
        v2 = [4, 5, 6]
        expected_distance = 0.974631
        calculated_distance = Distance.COSINE.between(v1, v2)
        assert_almost_equal(calculated_distance, expected_distance, decimal=6)

    def test_dot_distance_between(self):
        v1 = [1, 2, 3]
        v2 = [4, 5, 6]
        expected_distance = 32.0
        calculated_distance = Distance.DOT.between(v1, v2)
        assert_almost_equal(calculated_distance, expected_distance, decimal=6)

    def test_euclid_accept_score(self):
        self.assertTrue(Distance.EUCLID.accept_score(0.2, 0.3))
        self.assertFalse(Distance.EUCLID.accept_score(0.5, 0.3))

    def test_cosine_accept_score(self):
        self.assertTrue(Distance.COSINE.accept_score(0.8, 0.7))
        self.assertFalse(Distance.COSINE.accept_score(0.5, 0.7))

    def test_dot_accept_score(self):
        self.assertTrue(Distance.DOT.accept_score(0.5, 0.2))
        self.assertTrue(Distance.DOT.accept_score(0.5, 0.5))
        self.assertFalse(Distance.DOT.accept_score(0.5, 0.6))

    def test_euclid_sort(self):
        sorted_points = Distance.EUCLID.sort(self.points)
        expected_sorted_points = [
            Point(vector=[7, 8, 9], score=0.2),
            Point(vector=[4, 5, 6], score=0.5),
            Point(vector=[1, 2, 3], score=0.8)
        ]
        self.assertEqual(sorted_points, expected_sorted_points)

    def test_cosine_sort(self):
        sorted_points = Distance.COSINE.sort(self.points)
        expected_sorted_points = [
            Point(vector=[1, 2, 3], score=0.8),
            Point(vector=[4, 5, 6], score=0.5),
            Point(vector=[7, 8, 9], score=0.2)
        ]
        self.assertEqual(sorted_points, expected_sorted_points)

    def test_dot_sort(self):
        sorted_points = Distance.DOT.sort(self.points)
        expected_sorted_points = [
            Point(vector=[1, 2, 3], score=0.8),
            Point(vector=[4, 5, 6], score=0.5),
            Point(vector=[7, 8, 9], score=0.2)
        ]
        self.assertEqual(sorted_points, expected_sorted_points)

    def test_euclid_calculate_score(self):
        query_vector = self.query_vector
        point = Point(vector=[1, 2, 3], score=None)
        calculated_point = Distance.EUCLID.calculate_score(query_vector, point)
        expected_point = Point(vector=[1, 2, 3], score=2.23606798)
        assert_almost_equal(calculated_point.score, expected_point.score, decimal=6)

    def test_cosine_calculate_score(self):
        point = Point(vector=[1, 2, 3], score=None)
        calculated_point = Distance.COSINE.calculate_score(self.query_vector, point)
        expected_point = Point(vector=[1, 2, 3], score=0.9258201)
        assert_almost_equal(calculated_point.score, expected_point.score, decimal=6)

    def test_dot_calculate_score(self):
        point = Point(vector=[1, 2, 3], score=None)
        calculated_point = Distance.DOT.calculate_score(self.query_vector, point)
        expected_point = Point(vector=[1, 2, 3], score=6)
        assert_almost_equal(calculated_point.score, expected_point.score, decimal=6)

    def test_euclid_calculate_scores(self):
        calculated_points = Distance.EUCLID.calculate_scores(self.query_vector, self.points)
        expected_points = [
            Point(vector=[1, 2, 3], score=2.23606798),
            Point(vector=[4, 5, 6], score=7.07106781),
            Point(vector=[7, 8, 9], score=12.20655562)
        ]
        for calculated, expected in zip(calculated_points, expected_points):
            assert_almost_equal(calculated.score, expected.score, decimal=6)

    def test_cosine_calculate_scores(self):
        calculated_points = Distance.COSINE.calculate_scores(self.query_vector, self.points)
        expected_points = [
            Point(vector=[1, 2, 3], score=0.9258200997725514),
            Point(vector=[4, 5, 6], score=0.9869275424396534),
            Point(vector=[7, 8, 9], score=0.994832006747614),
        ]
        for calculated, expected in zip(calculated_points, expected_points):
            assert_almost_equal(calculated.score, expected.score, decimal=6)

    def test_dot_calculate_scores(self):
        calculated_points = Distance.DOT.calculate_scores(self.query_vector, self.points)
        expected_points = [
            Point(vector=[1, 2, 3], score=6),
            Point(vector=[4, 5, 6], score=15),
            Point(vector=[7, 8, 9], score=24)
        ]
        for calculated, expected in zip(calculated_points, expected_points):
            assert_almost_equal(calculated.score, expected.score, decimal=6)

    def test_euclid_filter_no_threshold(self):
        limit = 3
        filtered_points = Distance.EUCLID.filter(self.points, limit)
        expected_points = self.points
        self.assertEqual(filtered_points, expected_points)

    def test_cosine_filter_no_threshold(self):
        limit = 3
        filtered_points = Distance.COSINE.filter(self.points, limit)
        expected_points = self.points
        self.assertEqual(filtered_points, expected_points)

    def test_dot_filter_no_threshold(self):
        limit = 3
        filtered_points = Distance.DOT.filter(self.points, limit)
        expected_points = self.points
        self.assertEqual(filtered_points, expected_points)

    def test_euclid_filter_with_threshold(self):
        limit = 3
        score_threshold = 10.0
        filtered_points = Distance.EUCLID.filter(self.points, limit, score_threshold)
        expected_points = self.points
        self.assertEqual(filtered_points, expected_points)

    def test_cosine_filter_with_threshold(self):
        limit = 3
        score_threshold = 0.6
        filtered_points = Distance.COSINE.filter(self.points, limit, score_threshold)
        expected_points = [
            Point(vector=[1, 2, 3], score=0.8)
        ]
        self.assertEqual(filtered_points, expected_points)

    def test_dot_filter_with_threshold(self):
        limit = 3
        score_threshold = 0.5
        filtered_points = Distance.DOT.filter(self.points, limit, score_threshold)
        expected_points = [
            Point(vector=[1, 2, 3], score=0.8),
            Point(vector=[4, 5, 6], score=0.5),
        ]
        self.assertEqual(filtered_points, expected_points)


if __name__ == '__main__':
    unittest.main()
