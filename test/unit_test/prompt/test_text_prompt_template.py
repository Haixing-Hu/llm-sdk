# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import unittest

from llmsdk.prompt import TextPromptTemplate


class TestTextPromptTemplate(unittest.TestCase):

    def test_constructor(self):
        t1 = "Write an advertisement for the product {product}."
        p1 = TextPromptTemplate(template=t1)
        self.assertEqual(t1, p1.template)

        t2 = "What is {concept}."
        p2 = TextPromptTemplate(template=t2)
        self.assertEqual(t2, p2.template)

    def test_format(self):
        t1 = "This is a sample prompt {f1} and {f2}."
        p1 = TextPromptTemplate(template=t1)
        s1 = p1.format(f1="v1", f2="v2")
        self.assertEqual("This is a sample prompt v1 and v2.", s1)

        t2 = "This is another sample prompt {f1} and {f2} and {f3}."
        p2 = TextPromptTemplate(template=t2)
        s2 = p2.format(f1="v1", f2="v2", f3="v3")
        self.assertEqual("This is another sample prompt v1 and v2 and v3.", s2)

        t3 = "This is the third sample prompt {f1} and {f2} and {f3}."
        p3 = TextPromptTemplate(template=t3)
        with self.assertRaises(KeyError):
            p3.format(f1="v1", f2="v2")

        s3 = p3.format(f1="v1", f2="v2", f3="v3", f4="v4")
        self.assertEqual("This is the third sample prompt v1 and v2 and v3.", s3)


if __name__ == '__main__':
    unittest.main()
