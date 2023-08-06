# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
import unittest

from llmsdk.common import Metadata


class TestMetadata(unittest.TestCase):

    def test_constructor(self):
        m1 = Metadata()
        self.assertEqual({}, m1.data)

        m2 = Metadata({"a": 1, "b": 2})
        self.assertEqual({"a": 1, "b": 2}, m2.data)

    def test_set_item(self):
        m1 = Metadata()
        m1["a"] = 1
        m1["a"] = "abc"
        m1["b"] = 0.1

    def test_set_item_with_unsupported_type(self):
        m1 = Metadata()
        with self.assertRaises(ValueError):
            m1["c"] = True

    def test_has(self):
        m1 = Metadata()
        m1["a"] = 1
        m1["b"] = "b"
        m1["c"] = 0.1
        self.assertEqual(True, m1.has_value_of_type("a", int))
        self.assertEqual(False, m1.has_value_of_type("a", float))
        self.assertEqual(False, m1.has_value_of_type("a", str))
        self.assertEqual(True, m1.has_value_of_type("b", str))
        self.assertEqual(False, m1.has_value_of_type("b", int))
        self.assertEqual(False, m1.has_value_of_type("b", float))
        self.assertEqual(True, m1.has_value_of_type("c", float))
        self.assertEqual(False, m1.has_value_of_type("c", int))
        self.assertEqual(False, m1.has_value_of_type("c", str))
        self.assertEqual(False, m1.has_value_of_type("d", int))
        self.assertEqual(False, m1.has_value_of_type("d", float))
        self.assertEqual(False, m1.has_value_of_type("d", str))


if __name__ == '__main__':
    unittest.main()
