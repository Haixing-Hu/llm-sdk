# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import unittest
from numpy.testing import assert_almost_equal

from llmsdk.util.math_utils import euclid_distance, dot_distance, cosine_distance


class TestMathUtils(unittest.TestCase):
    def test_euclid_distance(self):
        v1 = [1, 2, 3]
        v2 = [4, 5, 6]
        expected_distance = 5.196152
        calculated_distance = euclid_distance(v1, v2)
        assert_almost_equal(calculated_distance, expected_distance, decimal=6)

    def test_dot_distance(self):
        v1 = [1, 2, 3]
        v2 = [4, 5, 6]
        expected_distance = 32.0
        calculated_distance = dot_distance(v1, v2)
        assert_almost_equal(calculated_distance, expected_distance, decimal=6)

    def test_cosine_distance(self):
        v1 = [1, 2, 3]
        v2 = [4, 5, 6]
        expected_distance = 0.974631
        calculated_distance = cosine_distance(v1, v2)
        assert_almost_equal(calculated_distance, expected_distance, decimal=6)


if __name__ == '__main__':
    unittest.main()
