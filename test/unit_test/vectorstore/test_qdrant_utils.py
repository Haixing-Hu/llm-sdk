# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import unittest

from qdrant_client.http import models

from llmsdk.common import Point
from llmsdk.generator import Uuid4Generator
from llmsdk.vectorstore.qdrant_utils import (
    to_qdrant_point,
    to_local_point,
    criterion_to_filter,
    simple_criterion_to_condition,
    composed_criterion_to_filter,
)
from llmsdk.criterion import (
    equal,
    not_equal,
    less,
    less_equal,
    greater,
    greater_equal,
    is_in,
    not_in,
    like,
    not_like,
    is_null,
    not_null,
    ComposedCriterionBuilder,
    Relation
)


class TestQdrantUtils(unittest.TestCase):

    def test_point_to_point_struct(self):
        id_generator = Uuid4Generator()

        p1 = Point([1.0, 2.0], {"page": 1}, id="id-1", score=123)
        s1 = to_qdrant_point(p1, id_generator)
        self.assertEqual([1.0, 2.0],  s1.vector)
        self.assertEqual({"page": 1}, s1.payload)
        self.assertEqual("id-1", s1.id)

        p2 = Point([2.0, 3.0], {"page": 2})
        self.assertIsNone(p2.id)
        s2 = to_qdrant_point(p2, id_generator)
        self.assertEqual([2.0, 3.0],  s2.vector)
        self.assertEqual({"page": 2}, s2.payload)
        self.assertIsNotNone(p2.id)
        self.assertIsNotNone(s2.id)
        self.assertEqual(p2.id, s2.id)

    def test_scored_point_to_point(self):
        s1 = models.ScoredPoint(id="id-1",
                                version=1,
                                score=3.14,
                                payload={"page": 1},
                                vector=[1.0, 2.0])
        p1 = to_local_point(s1)
        self.assertEqual("id-1", p1.id)
        self.assertEqual({"page": 1}, p1.metadata)
        self.assertEqual([1.0, 2.0], p1.vector)
        self.assertEqual(3.14, p1.score)

    def test_simple_criterion_to_filter(self):
        c1 = equal("f1", "v1")
        r1 = simple_criterion_to_condition(c1)
        self.assertIsInstance(r1, models.FieldCondition)
        self.assertEqual("f1", r1.key)
        self.assertIsInstance(r1.match, models.MatchValue)
        self.assertEqual("v1", r1.match.value)

        c2 = not_equal("f2", "v2")
        r2 = simple_criterion_to_condition(c2)
        self.assertIsInstance(r2, models.Filter)
        self.assertEqual(1, len(r2.must_not))
        self.assertIsInstance(r2.must_not[0], models.FieldCondition)
        self.assertEqual("f2", r2.must_not[0].key)
        self.assertIsInstance(r2.must_not[0].match, models.MatchValue)
        self.assertEqual("v2", r2.must_not[0].match.value)

        c3 = less("f3", 100)
        r3 = simple_criterion_to_condition(c3)
        self.assertIsInstance(r3, models.FieldCondition)
        self.assertEqual("f3", r3.key)
        self.assertIsInstance(r3.range, models.Range)
        self.assertEqual(100, r3.range.lt)
        self.assertIsNone(r3.range.lte)
        self.assertIsNone(r3.range.gt)
        self.assertIsNone(r3.range.gte)

        c4 = less_equal("f4", 100)
        r4 = simple_criterion_to_condition(c4)
        self.assertIsInstance(r4, models.FieldCondition)
        self.assertEqual("f4", r4.key)
        self.assertIsInstance(r4.range, models.Range)
        self.assertEqual(100, r4.range.lte)
        self.assertIsNone(r4.range.lt)
        self.assertIsNone(r4.range.gt)
        self.assertIsNone(r4.range.gte)

        c5 = greater("f5", 100)
        r5 = simple_criterion_to_condition(c5)
        self.assertIsInstance(r5, models.FieldCondition)
        self.assertEqual("f5", r5.key)
        self.assertIsInstance(r5.range, models.Range)
        self.assertEqual(100, r5.range.gt)
        self.assertIsNone(r5.range.lte)
        self.assertIsNone(r5.range.lt)
        self.assertIsNone(r5.range.gte)

        c6 = greater_equal("f6", 100)
        r6 = simple_criterion_to_condition(c6)
        self.assertIsInstance(r6, models.FieldCondition)
        self.assertEqual("f6", r6.key)
        self.assertIsInstance(r6.range, models.Range)
        self.assertEqual(100, r6.range.gte)
        self.assertIsNone(r6.range.lt)
        self.assertIsNone(r6.range.gt)
        self.assertIsNone(r6.range.lte)

        c7 = is_in("f7", ["a", "b", "c"])
        r7 = simple_criterion_to_condition(c7)
        self.assertIsInstance(r7, models.FieldCondition)
        self.assertEqual("f7", r7.key)
        self.assertIsInstance(r7.match, models.MatchAny)
        self.assertEqual(["a", "b", "c"], r7.match.any)

        c8 = not_in("f8", ["a", "b", "c"])
        r8 = simple_criterion_to_condition(c8)
        self.assertIsInstance(r8, models.Filter)
        self.assertEqual(1, len(r8.must_not))
        self.assertIsInstance(r8.must_not[0], models.FieldCondition)
        self.assertEqual("f8", r8.must_not[0].key)
        self.assertIsInstance(r8.must_not[0].match, models.MatchAny)
        self.assertEqual(["a", "b", "c"], r8.must_not[0].match.any)

        c9 = like("f9", "v9")
        r9 = simple_criterion_to_condition(c9)
        self.assertIsInstance(r9, models.FieldCondition)
        self.assertEqual("f9", r9.key)
        self.assertIsInstance(r9.match, models.MatchText)
        self.assertEqual("v9", r9.match.text)

        c10 = not_like("f10", "v10")
        r10 = simple_criterion_to_condition(c10)
        self.assertIsInstance(r10, models.Filter)
        self.assertEqual(1, len(r10.must_not))
        self.assertIsInstance(r10.must_not[0], models.FieldCondition)
        self.assertEqual("f10", r10.must_not[0].key)
        self.assertIsInstance(r10.must_not[0].match, models.MatchText)
        self.assertEqual("v10", r10.must_not[0].match.text)

        c11 = is_null("f11")
        r11 = simple_criterion_to_condition(c11)
        self.assertIsInstance(r11, models.IsNullCondition)
        self.assertIsInstance(r11.is_null, models.PayloadField)
        self.assertEqual("f11", r11.is_null.key)

        c12 = not_null("f12")
        r12 = simple_criterion_to_condition(c12)
        self.assertIsInstance(r12, models.Filter)
        self.assertEqual(1, len(r12.must_not))
        self.assertIsInstance(r12.must_not[0], models.IsNullCondition)
        self.assertIsInstance(r12.must_not[0].is_null, models.PayloadField)
        self.assertEqual("f12", r12.must_not[0].is_null.key)

    def test_composed_criterion_to_filter(self):
        c1 = ComposedCriterionBuilder(Relation.AND)\
            .equal("f1", "v1")\
            .not_equal("f2", "v2")\
            .less("f3", 100)\
            .not_in("f4", ["a", "b", "c"])\
            .build()
        r1 = composed_criterion_to_filter(c1)
        self.assertIsInstance(r1, models.Filter)
        self.assertEqual(4, len(r1.must))
        e0 = simple_criterion_to_condition(equal("f1", "v1"))
        self.assertEqual(e0, r1.must[0])
        e1 = simple_criterion_to_condition(not_equal("f2", "v2"))
        self.assertEqual(e1, r1.must[1])
        e2 = simple_criterion_to_condition(less("f3", 100))
        self.assertEqual(e2, r1.must[2])
        e3 = simple_criterion_to_condition(not_in("f4", ["a", "b", "c"]))
        self.assertEqual(e3, r1.must[3])

    def test_criterion_to_filter(self):
        c1 = equal("f1", "v1")
        r1 = criterion_to_filter(c1)
        self.assertIsInstance(r1, models.Filter)
        self.assertEqual(1, len(r1.must))
        self.assertIsInstance(r1.must[0], models.FieldCondition)
        self.assertEqual("f1", r1.must[0].key)
        self.assertIsInstance(r1.must[0].match, models.MatchValue)
        self.assertEqual("v1", r1.must[0].match.value)

        c2 = ComposedCriterionBuilder(Relation.AND) \
            .equal("f1", "v1") \
            .not_equal("f2", "v2") \
            .less("f3", 100) \
            .not_in("f4", ["a", "b", "c"]) \
            .build()
        r2 = criterion_to_filter(c2)
        self.assertIsInstance(r2, models.Filter)
        self.assertEqual(4, len(r2.must))
        e0 = simple_criterion_to_condition(equal("f1", "v1"))
        self.assertEqual(e0, r2.must[0])
        e1 = simple_criterion_to_condition(not_equal("f2", "v2"))
        self.assertEqual(e1, r2.must[1])
        e2 = simple_criterion_to_condition(less("f3", 100))
        self.assertEqual(e2, r2.must[2])
        e3 = simple_criterion_to_condition(not_in("f4", ["a", "b", "c"]))
        self.assertEqual(e3, r2.must[3])


if __name__ == '__main__':
    unittest.main()
