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


if __name__ == '__main__':
    unittest.main()
