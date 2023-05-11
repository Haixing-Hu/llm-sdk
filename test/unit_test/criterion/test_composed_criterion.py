# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import unittest
from dataclasses import FrozenInstanceError

from llmsdk.criterion import SimpleCriterion, ComposedCriterion, Operator, Relation


class TestComposedCriterion(unittest.TestCase):

    def test_constructor(self):
        s1 = SimpleCriterion("f1", Operator.EQUAL, "v1")
        self.assertEqual("f1", s1.property)
        self.assertEqual(Operator.EQUAL, s1.operator)
        self.assertEqual("v1", s1.value)

        s2 = SimpleCriterion("f2.ff2.fff2", Operator.LESS_EQUAL, 100)
        self.assertEqual("f2.ff2.fff2", s2.property)
        self.assertEqual(Operator.LESS_EQUAL, s2.operator)
        self.assertEqual(100, s2.value)

        s3 = SimpleCriterion("f3.e3", Operator.IS_NULL)
        self.assertEqual("f3.e3", s3.property)
        self.assertEqual(Operator.IS_NULL, s3.operator)
        self.assertIsNone(s3.value)

        c1 = ComposedCriterion(Relation.AND, [s1, s2, s3])
        self.assertEqual(Relation.AND, c1.relation)
        self.assertEqual([s1, s2, s3], c1.criteria)

    def test_immutable(self):
        s1 = SimpleCriterion("f1", Operator.EQUAL, "v1")
        s2 = SimpleCriterion("f2.ff2.fff2", Operator.LESS_EQUAL, 100)
        s3 = SimpleCriterion("f3.e3", Operator.IS_NULL)
        c1 = ComposedCriterion(Relation.AND, [s1, s2, s3])
        with self.assertRaises(FrozenInstanceError):
            c1.relation = Relation.OR
        with self.assertRaises(FrozenInstanceError):
            c1.criteria = [s1]


if __name__ == '__main__':
    unittest.main()
