# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import unittest

from llmsdk.common import Example


class TestExample(unittest.TestCase):

    def test_constructor(self):
        ex1 = Example("input1", "output1")
        self.assertIsNone(ex1.id)
        self.assertEqual("input1", ex1.input)
        self.assertEqual("output1", ex1.output)

        ex2 = Example(id="id2", input="input2", output="output2")
        self.assertEqual("id2", ex2.id)
        self.assertEqual("input2", ex2.input)
        self.assertEqual("output2", ex2.output)

    def test_eq(self):
        # two examples are equal if their IDs are equal
        f1 = Example(id="faq-1", input="input1", output="output1", score=0.1)
        f2 = Example(id="faq-1", input="input1", output="output1", score=0.2)
        self.assertEqual(f1, f2)

        f3 = Example(id="faq-1", input="input1", output="output2", score=0.2)
        self.assertNotEqual(f1, f3)


if __name__ == '__main__':
    unittest.main()
