# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import unittest
from dataclasses import FrozenInstanceError

from llmsdk.criterion import SimpleCriterion, Operator


class TestSimpleCriterion(unittest.TestCase):

    def test_constructor(self):
        c1 = SimpleCriterion("f1", Operator.EQUAL, "v1")
        self.assertEqual("f1", c1.property)
        self.assertEqual(Operator.EQUAL, c1.operator)
        self.assertEqual("v1", c1.value)

        c2 = SimpleCriterion("f2.ff2.fff2", Operator.LESS_EQUAL, 100)
        self.assertEqual("f2.ff2.fff2", c2.property)
        self.assertEqual(Operator.LESS_EQUAL, c2.operator)
        self.assertEqual(100, c2.value)

        c3 = SimpleCriterion("f3.e3", Operator.IS_NULL)
        self.assertEqual("f3.e3", c3.property)
        self.assertEqual(Operator.IS_NULL, c3.operator)
        self.assertIsNone(c3.value)

    def test_immutable(self):
        c1 = SimpleCriterion("f1", Operator.EQUAL, "v1")
        self.assertEqual("f1", c1.property)
        self.assertEqual(Operator.EQUAL, c1.operator)
        self.assertEqual("v1", c1.value)
        with self.assertRaises(FrozenInstanceError):
            c1.field = "f2"
        with self.assertRaises(FrozenInstanceError):
            c1.operator = Operator.LESS_EQUAL
        with self.assertRaises(FrozenInstanceError):
            c1.value = "v2"


if __name__ == '__main__':
    unittest.main()
