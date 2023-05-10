# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import unittest

from llmsdk.common import Point


class TestPoint(unittest.TestCase):

    def test_constructor(self):
        p1 = Point()
        self.assertEqual([], p1.vector)
        self.assertEqual({}, p1.metadata)
        self.assertIsNone(p1.id)
        self.assertIsNone(p1.score)

        p2 = Point([1.1, 2.2])
        self.assertEqual([1.1, 2.2], p2.vector)
        self.assertEqual({}, p2.metadata)
        self.assertIsNone(p2.id)
        self.assertIsNone(p2.score)

        p3 = Point([1.1, 2.2, 3.3], {"name": "p3"})
        self.assertEqual([1.1, 2.2, 3.3], p3.vector)
        self.assertEqual({"name": "p3"}, p3.metadata)
        self.assertIsNone(p3.id)
        self.assertIsNone(p3.score)

        p4 = Point([1.1, 2.2, 3.3, 4.4], {"name": "p4"}, id="p4.id")
        self.assertEqual([1.1, 2.2, 3.3, 4.4], p4.vector)
        self.assertEqual({"name": "p4"}, p4.metadata)
        self.assertEqual("p4.id", p4.id)
        self.assertIsNone(p4.score)

        p5 = Point([1.1, 2.2, 3.3, 4.4, 5.5], {"name": "p5"}, id="p5.id", score=1.1)
        self.assertEqual([1.1, 2.2, 3.3, 4.4, 5.5], p5.vector)
        self.assertEqual({"name": "p5"}, p5.metadata)
        self.assertEqual("p5.id", p5.id)
        self.assertEqual(1.1, p5.score)


if __name__ == '__main__':
    unittest.main()
