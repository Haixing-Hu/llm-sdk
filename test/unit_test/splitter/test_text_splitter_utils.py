# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
import unittest

from llmsdk.splitter.text_splitter_utils import (
    combine_splits,
)


class TestTextSplitterUtils(unittest.TestCase):

    def test_combine_splits(self):
        r1 = combine_splits(splits=["foo", "bar", "baz", "123"],
                            separator=" ",
                            chunk_size=7,
                            chunk_overlap=3,
                            length_function=len)
        self.assertEqual(["foo bar", "bar baz", "baz 123"], r1)

        r2 = combine_splits(splits=["foo", "bar"],
                            separator=" ",
                            chunk_size=2,
                            chunk_overlap=0,
                            length_function=len)
        self.assertEqual(["foo", "bar"], r2)

        r3 = combine_splits(splits=["f", "b"],
                            separator=" ",
                            chunk_size=2,
                            chunk_overlap=0,
                            length_function=len)
        self.assertEqual(["f", "b"], r3)

        r4 = combine_splits(splits=["foo", "bar", "baz", "a", "a"],
                            separator=" ",
                            chunk_size=3,
                            chunk_overlap=1,
                            length_function=len)
        self.assertEqual(["foo", "bar", "baz", "a a"], r4)

        r5 = combine_splits(splits=["a", "a", "foo", "bar", "baz"],
                            separator=" ",
                            chunk_size=3,
                            chunk_overlap=1,
                            length_function=len)
        self.assertEqual(["a a", "foo", "bar", "baz"], r5)

        r6 = combine_splits(splits=["foo", "bar", "baz", "123"],
                            separator=" ",
                            chunk_size=1,
                            chunk_overlap=1,
                            length_function=len)
        self.assertEqual(["foo", "bar", "baz", "123"], r6)

        r7 = combine_splits(splits=["foo", "bar", "baz"],
                            separator=" ",
                            chunk_size=9,
                            chunk_overlap=2,
                            length_function=len)
        self.assertEqual(["foo bar", "baz"], r7)

        with self.assertRaises(ValueError):
            combine_splits(splits=["a", "a", "foo", "bar", "baz"],
                           separator=" ",
                           chunk_size=2,
                           chunk_overlap=4,
                           length_function=len)


if __name__ == '__main__':
    unittest.main()
