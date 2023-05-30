# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import unittest

from llmsdk.criterion import (
    SimpleCriterion,
    ComposedCriterion,
    Operator,
    Relation,
    ComposedCriterionBuilder,
)


class TestComposedCriterionBuilder(unittest.TestCase):

    def test_build(self):
        s1 = SimpleCriterion("f1", Operator.EQUAL, "v1")
        s2 = SimpleCriterion("f2.ff2.fff2", Operator.LESS_EQUAL, 100)
        s3 = SimpleCriterion("f3.e3", Operator.IS_NULL)
        c1 = ComposedCriterion(Relation.AND, [s1, s2, s3])
        c2 = ComposedCriterionBuilder(Relation.AND)\
            .equal("f1", "v1")\
            .less_equal("f2.ff2.fff2", 100)\
            .is_null("f3.e3")\
            .build()
        self.assertEqual(c1, c2)


if __name__ == '__main__':
    unittest.main()
