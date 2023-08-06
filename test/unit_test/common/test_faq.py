# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
import unittest

from llmsdk.common import Faq, Document, Metadata


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

    def test_eq(self):
        # two FAQs are equal if their IDs are equal
        f1 = Faq(id="faq-1", question="question", answer="answer", score=0.1)
        f2 = Faq(id="faq-1", question="question", answer="answer", score=0.2)
        self.assertEqual(f1, f2)

    def test_from_faq(self):
        e1 = Faq("question1", "answer1", id="faq-1")
        self.assertEqual("faq-1", e1.id)
        self.assertEqual("question1", e1.question)
        self.assertEqual("answer1", e1.answer)
        d1 = Faq.to_document(e1)
        self.assertEqual(2, len(d1))
        self.assertEqual("faq-1-question", d1[0].id)
        self.assertEqual("question1", d1[0].content)
        self.assertEqual(Metadata({
            "__faq_id__": "faq-1",
            "__faq_question__": "question1",
            "__faq_answer__": "answer1",
            "__faq_property__": "question",
        }), d1[0].metadata)
        self.assertIsNone(d1[0].score)
        self.assertEqual("faq-1-answer", d1[1].id)
        self.assertEqual("answer1", d1[1].content)
        self.assertEqual(Metadata({
            "__faq_id__": "faq-1",
            "__faq_question__": "question1",
            "__faq_answer__": "answer1",
            "__faq_property__": "answer",
        }), d1[1].metadata)
        self.assertIsNone(d1[1].score)

        e2 = Faq("question2", "answer2", id="faq-2", score=0.7)
        self.assertEqual("faq-2", e2.id)
        self.assertEqual("question2", e2.question)
        self.assertEqual("answer2", e2.answer)
        d2 = Faq.to_document(e2)
        self.assertEqual(2, len(d2))
        self.assertEqual("faq-2-question", d2[0].id)
        self.assertEqual("question2", d2[0].content)
        self.assertEqual(Metadata({
            "__faq_id__": "faq-2",
            "__faq_question__": "question2",
            "__faq_answer__": "answer2",
            "__faq_property__": "question",
        }), d2[0].metadata)
        self.assertEqual(0.7, d2[0].score)
        self.assertEqual("faq-2-answer", d2[1].id)
        self.assertEqual("answer2", d2[1].content)
        self.assertEqual(Metadata({
            "__faq_id__": "faq-2",
            "__faq_question__": "question2",
            "__faq_answer__": "answer2",
            "__faq_property__": "answer",
        }), d2[1].metadata)
        self.assertEqual(0.7, d2[1].score)


if __name__ == '__main__':
    unittest.main()
