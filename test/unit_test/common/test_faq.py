# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import unittest

from llmsdk.common import Faq


class TestFaq(unittest.TestCase):

    def test_constructor(self):
        f1 = Faq("question1", "answer1")
        self.assertIsNone(f1.id)
        self.assertEqual("question1", f1.question)
        self.assertEqual("answer1", f1.answer)

        f2 = Faq(id="id2", question="question2", answer="answer2")
        self.assertEqual("id2", f2.id)
        self.assertEqual("question2", f2.question)
        self.assertEqual("answer2", f2.answer)


if __name__ == '__main__':
    unittest.main()
