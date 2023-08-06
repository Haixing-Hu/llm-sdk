# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
import unittest

from llmsdk.common import Document, Metadata, Example, Faq
from llmsdk.embedding import MockEmbedding


class TestDocument(unittest.TestCase):

    def test_constructor(self):
        doc1 = Document("doc1")
        self.assertEqual("doc1", doc1.content)
        self.assertIsNotNone(doc1.metadata)
        self.assertEqual({}, doc1.metadata)

        meta2 = Metadata({"x": 1, "y": 2})
        doc2 = Document("doc2", metadata=meta2)
        self.assertEqual("doc2", doc2.content)
        self.assertEqual(meta2, doc2.metadata)

        self.assertEqual(1, meta2["x"])
        doc2.metadata["x"] = 3
        self.assertEqual(3, doc2.metadata["x"])
        # the passed constructor argument is also change
        self.assertEqual(3, meta2["x"])

    def test_to_point(self):
        texts = ["content1", "content2"]
        embedding = MockEmbedding()
        vectors = embedding.embed_texts(texts)

        d0 = Document(id="001",
                      content=texts[0],
                      metadata=Metadata({"f1": "v1", "f2": "v2"}))

        p0 = Document.to_point(d0, vectors[0])
        self.assertIsNone(p0.id)
        self.assertEqual(vectors[0], p0.vector)
        m0 = Metadata({
            "f1": "v1",
            "f2": "v2",
            Document.DOCUMENT_ID_ATTRIBUTE: d0.id,
            Document.DOCUMENT_CONTENT_ATTRIBUTE: d0.content,
        })
        self.assertEqual(m0, p0.metadata)
        self.assertIsNone(p0.score)

        d1 = Document(id="001",
                      content=texts[1],
                      metadata=Metadata({"f1": "v1", "f2": "v2", "f3": "v3"}))

        p1 = Document.to_point(d1, vectors[1])
        self.assertIsNone(p1.id)
        self.assertEqual(vectors[1], p1.vector)
        m1 = Metadata({
            "f1": "v1",
            "f2": "v2",
            "f3": "v3",
            Document.DOCUMENT_ID_ATTRIBUTE: d1.id,
            Document.DOCUMENT_CONTENT_ATTRIBUTE: d1.content,
        })
        self.assertEqual(m1, p1.metadata)
        self.assertIsNone(p1.score)

    def test_from_point(self):
        texts = ["content1", "content2"]
        embedding = MockEmbedding()
        vectors = embedding.embed_texts(texts)

        d0 = Document(id="001",
                      content=texts[0],
                      metadata=Metadata({"f1": "v1", "f2": "v2"}))
        p0 = Document.to_point(d0, vectors[0])
        a0 = Document.from_point(p0)
        self.assertEqual(d0, a0)

        d1 = Document(id="001",
                      content=texts[1],
                      metadata=Metadata({"f1": "v1", "f2": "v2", "f3": "v3"}))
        p1 = Document.to_point(d1, vectors[1])
        a1 = Document.from_point(p1)
        self.assertEqual(d1, a1)

    def test_from_example(self):
        e1 = Example("input1", "output1", id="example-1")
        self.assertEqual("example-1", e1.id)
        self.assertEqual("input1", e1.input)
        self.assertEqual("output1", e1.output)
        d1 = Document.from_example(e1)
        self.assertEqual(2, len(d1))
        self.assertEqual("example-1-input", d1[0].id)
        self.assertEqual("input1", d1[0].content)
        self.assertEqual(Metadata({
            "__example_id__": "example-1",
            "__example_input__": "input1",
            "__example_output__": "output1",
            "__example_property__": "input",
        }), d1[0].metadata)
        self.assertIsNone(d1[0].score)
        self.assertEqual("example-1-output", d1[1].id)
        self.assertEqual("output1", d1[1].content)
        self.assertEqual(Metadata({
            "__example_id__": "example-1",
            "__example_input__": "input1",
            "__example_output__": "output1",
            "__example_property__": "output",
        }), d1[1].metadata)
        self.assertIsNone(d1[1].score)

        e2 = Example("input2", "output2", id="example-2", score=0.7)
        self.assertEqual("example-2", e2.id)
        self.assertEqual("input2", e2.input)
        self.assertEqual("output2", e2.output)
        d2 = Document.from_example(e2)
        self.assertEqual(2, len(d2))
        self.assertEqual("example-2-input", d2[0].id)
        self.assertEqual("input2", d2[0].content)
        self.assertEqual(Metadata({
            "__example_id__": "example-2",
            "__example_input__": "input2",
            "__example_output__": "output2",
            "__example_property__": "input",
        }), d2[0].metadata)
        self.assertEqual(0.7, d2[0].score)
        self.assertEqual("example-2-output", d2[1].id)
        self.assertEqual("output2", d2[1].content)
        self.assertEqual(Metadata({
            "__example_id__": "example-2",
            "__example_input__": "input2",
            "__example_output__": "output2",
            "__example_property__": "output",
        }), d2[1].metadata)
        self.assertEqual(0.7, d2[1].score)


if __name__ == '__main__':
    unittest.main()
