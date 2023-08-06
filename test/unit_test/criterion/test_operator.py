# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
import unittest

from llmsdk.criterion import Operator


class TestOperator(unittest.TestCase):

    def test_equal(self):
        self.assertTrue(Operator.EQUAL.test(5, 5))
        self.assertFalse(Operator.EQUAL.test(5, 10))
        self.assertFalse(Operator.EQUAL.test(5, "5"))

    def test_not_equal(self):
        self.assertTrue(Operator.NOT_EQUAL.test(5, 10))
        self.assertFalse(Operator.NOT_EQUAL.test(5, 5))
        self.assertTrue(Operator.NOT_EQUAL.test(5, "5"))

    def test_less(self):
        self.assertTrue(Operator.LESS.test(5, 10))
        self.assertFalse(Operator.LESS.test(10, 5))
        self.assertFalse(Operator.LESS.test(5, 5))

    def test_less_equal(self):
        self.assertTrue(Operator.LESS_EQUAL.test(5, 10))
        self.assertFalse(Operator.LESS_EQUAL.test(10, 5))
        self.assertTrue(Operator.LESS_EQUAL.test(5, 5))

    def test_greater(self):
        self.assertTrue(Operator.GREATER.test(10, 5))
        self.assertFalse(Operator.GREATER.test(5, 10))
        self.assertFalse(Operator.GREATER.test(5, 5))

    def test_greater_equal(self):
        self.assertTrue(Operator.GREATER_EQUAL.test(10, 5))
        self.assertFalse(Operator.GREATER_EQUAL.test(5, 10))
        self.assertTrue(Operator.GREATER_EQUAL.test(5, 5))

    def test_in(self):
        self.assertTrue(Operator.IN.test(5, [1, 2, 3, 4, 5]))
        self.assertFalse(Operator.IN.test(5, [1, 2, 3, 4]))
        with self.assertRaises(ValueError):
            Operator.IN.test(5, "12345")

    def test_not_in(self):
        self.assertTrue(Operator.NOT_IN.test(5, [1, 2, 3, 4]))
        self.assertFalse(Operator.NOT_IN.test(5, [1, 2, 3, 4, 5]))
        with self.assertRaises(ValueError):
            Operator.NOT_IN.test(5, "12345")

    def test_like(self):
        self.assertTrue(Operator.LIKE.test("hello", "he%"))
        self.assertFalse(Operator.LIKE.test("hello", "hi%"))
        self.assertTrue(Operator.LIKE.test("hello", "h%o"))
        with self.assertRaises(ValueError):
            self.assertFalse(Operator.LIKE.test("hello", 123))

    def test_not_like(self):
        self.assertTrue(Operator.NOT_LIKE.test("hello", "hi%"))
        self.assertFalse(Operator.NOT_LIKE.test("hello", "he%"))
        self.assertFalse(Operator.NOT_LIKE.test("hello", "h%o"))
        with self.assertRaises(ValueError):
            self.assertTrue(Operator.NOT_LIKE.test("hello", 123))

    def test_is_null(self):
        self.assertTrue(Operator.IS_NULL.test(None))
        self.assertFalse(Operator.IS_NULL.test(5))

    def test_not_null(self):
        self.assertFalse(Operator.NOT_NULL.test(None))
        self.assertTrue(Operator.NOT_NULL.test(5))
        self.assertTrue(Operator.NOT_NULL.test("hello"))


if __name__ == '__main__':
    unittest.main()
