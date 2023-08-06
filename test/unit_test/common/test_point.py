# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
import unittest

from llmsdk.common import Point, Metadata, Document
from llmsdk.embedding import MockEmbedding


class TestPoint(unittest.TestCase):

    def test_constructor(self):
        p1 = Point()
        self.assertEqual([], p1.vector)
        self.assertEqual({}, p1.metadata)
        self.assertIsNone(p1.id)
        self.assertIsNone(p1.score)

        p2 = Point([1.1, 2.2])
        self.assertEqual([1.1, 2.2], p2.vector)
        self.assertEqual({}, p2.metadata)
        self.assertIsNone(p2.id)
        self.assertIsNone(p2.score)

        p3 = Point([1.1, 2.2, 3.3], Metadata({"name": "p3"}))
        self.assertEqual([1.1, 2.2, 3.3], p3.vector)
        self.assertEqual({"name": "p3"}, p3.metadata)
        self.assertIsNone(p3.id)
        self.assertIsNone(p3.score)

        p4 = Point([1.1, 2.2, 3.3, 4.4], Metadata({"name": "p4"}), id="p4.id")
        self.assertEqual([1.1, 2.2, 3.3, 4.4], p4.vector)
        self.assertEqual({"name": "p4"}, p4.metadata)
        self.assertEqual("p4.id", p4.id)
        self.assertIsNone(p4.score)

        p5 = Point([1.1, 2.2, 3.3, 4.4, 5.5], Metadata({"name": "p5"}),
                   id="p5.id", score=1.1)
        self.assertEqual([1.1, 2.2, 3.3, 4.4, 5.5], p5.vector)
        self.assertEqual({"name": "p5"}, p5.metadata)
        self.assertEqual("p5.id", p5.id)
        self.assertEqual(1.1, p5.score)

    def test_metadata_has(self):
        p1 = Point()
        self.assertEqual(False, p1.metadata.has_value_of_type("a", int))

        p2 = Point([1.1, 2.2, 3.3], Metadata({"name": "p2"}))
        self.assertEqual(True, p2.metadata.has_value_of_type("name", str))
        self.assertEqual(False, p2.metadata.has_value_of_type("name", int))
        self.assertEqual(False, p2.metadata.has_value_of_type("name", float))

    def test_from_document(self):
        texts = ["content1", "content2"]
        embedding = MockEmbedding()
        vectors = embedding.embed_texts(texts)

        d0 = Document(id="001",
                      content=texts[0],
                      metadata=Metadata({"f1": "v1", "f2": "v2"}))

        p0 = Point.from_document(d0, vectors[0])
        self.assertIsNone(p0.id)
        self.assertEqual(vectors[0], p0.vector)
        m0 = Metadata({
            "f1": "v1",
            "f2": "v2",
            "__document_id__": d0.id,
            "__document_content__": d0.content,
        })
        self.assertEqual(m0, p0.metadata)
        self.assertIsNone(p0.score)

        d1 = Document(id="001",
                      content=texts[1],
                      metadata=Metadata({"f1": "v1", "f2": "v2", "f3": "v3"}))

        p1 = Point.from_document(d1, vectors[1])
        self.assertIsNone(p1.id)
        self.assertEqual(vectors[1], p1.vector)
        m1 = Metadata({
            "f1": "v1",
            "f2": "v2",
            "f3": "v3",
            "__document_id__": d1.id,
            "__document_content__": d1.content,
        })
        self.assertEqual(m1, p1.metadata)
        self.assertIsNone(p1.score)

    def test_to_document(self):
        texts = ["content1", "content2"]
        embedding = MockEmbedding()
        vectors = embedding.embed_texts(texts)

        d0 = Document(id="001",
                      content=texts[0],
                      metadata=Metadata({"f1": "v1", "f2": "v2"}))
        p0 = Point.from_document(d0, vectors[0])
        a0 = Point.to_document(p0)
        self.assertEqual(d0, a0)

        d1 = Document(id="001",
                      content=texts[1],
                      metadata=Metadata({"f1": "v1", "f2": "v2", "f3": "v3"}))
        p1 = Point.from_document(d1, vectors[1])
        a1 = Point.to_document(p1)
        self.assertEqual(d1, a1)


if __name__ == '__main__':
    unittest.main()
