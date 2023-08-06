# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
import unittest

from llmsdk.common import Example, Metadata


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

    def test_to_document(self):
        e1 = Example("input1", "output1", id="example-1")
        self.assertEqual("example-1", e1.id)
        self.assertEqual("input1", e1.input)
        self.assertEqual("output1", e1.output)
        d1 = Example.to_document(e1)
        self.assertEqual(2, len(d1))
        self.assertEqual("example-1-input", d1[0].id)
        self.assertEqual("input1", d1[0].content)
        self.assertEqual(Metadata({
            "__example_id__": "example-1",
            "__example_input__": "input1",
            "__example_output__": "output1",
            "__example_part__": "input",
            "__type__": "EXAMPLE",
        }), d1[0].metadata)
        self.assertIsNone(d1[0].score)
        self.assertEqual("example-1-output", d1[1].id)
        self.assertEqual("output1", d1[1].content)
        self.assertEqual(Metadata({
            "__example_id__": "example-1",
            "__example_input__": "input1",
            "__example_output__": "output1",
            "__example_part__": "output",
            "__type__": "EXAMPLE",
        }), d1[1].metadata)
        self.assertIsNone(d1[1].score)

        e2 = Example("input2", "output2", id="example-2", score=0.7)
        self.assertEqual("example-2", e2.id)
        self.assertEqual("input2", e2.input)
        self.assertEqual("output2", e2.output)
        d2 = Example.to_document(e2)
        self.assertEqual(2, len(d2))
        self.assertEqual("example-2-input", d2[0].id)
        self.assertEqual("input2", d2[0].content)
        self.assertEqual(Metadata({
            "__example_id__": "example-2",
            "__example_input__": "input2",
            "__example_output__": "output2",
            "__example_part__": "input",
            "__type__": "EXAMPLE",
        }), d2[0].metadata)
        self.assertEqual(0.7, d2[0].score)
        self.assertEqual("example-2-output", d2[1].id)
        self.assertEqual("output2", d2[1].content)
        self.assertEqual(Metadata({
            "__example_id__": "example-2",
            "__example_input__": "input2",
            "__example_output__": "output2",
            "__example_part__": "output",
            "__type__": "EXAMPLE",
        }), d2[1].metadata)
        self.assertEqual(0.7, d2[1].score)


if __name__ == '__main__':
    unittest.main()
