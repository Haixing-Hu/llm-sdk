# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import unittest

from llmsdk.common import Document, Metadata
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

        p0 = d0.to_point(vectors[0])
        self.assertEqual(d0.id, p0.id)
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

        p1 = d1.to_point(vectors[1])
        self.assertEqual(d1.id, p1.id)
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
        p0 = d0.to_point(vectors[0])
        a0 = Document.from_point(p0)
        self.assertEqual(d0, a0)

        d1 = Document(id="001",
                      content=texts[1],
                      metadata=Metadata({"f1": "v1", "f2": "v2", "f3": "v3"}))
        p1 = d1.to_point(vectors[1])
        a1 = Document.from_point(p1)
        self.assertEqual(d1, a1)


if __name__ == '__main__':
    unittest.main()
