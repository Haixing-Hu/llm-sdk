# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
import unittest

from llmsdk.common import Document, Metadata


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

    def test_from_record(self):
        record = {
            "id": "001",
            "name": "doc1",
            "spec": "10ml",
            "value": 1,
            "delta": 0.13,
        }
        docs = Document.from_record("id", record)
        self.assertEqual(4, len(docs))

        self.assertEqual("doc1", docs[0].content)
        self.assertEqual(Metadata({
            "id": "001",
            "name": "doc1",
            "spec": "10ml",
            "value": 1,
            "delta": 0.13,
            "__type__": "RECORD",
            "__record_field__": "name",
        }), docs[0].metadata)

        self.assertEqual("10ml", docs[1].content)
        self.assertEqual(Metadata({
            "id": "001",
            "name": "doc1",
            "spec": "10ml",
            "value": 1,
            "delta": 0.13,
            "__type__": "RECORD",
            "__record_field__": "spec",
        }), docs[1].metadata)

        self.assertEqual("1", docs[2].content)
        self.assertEqual(Metadata({
            "id": "001",
            "name": "doc1",
            "spec": "10ml",
            "value": 1,
            "delta": 0.13,
            "__type__": "RECORD",
            "__record_field__": "value",
        }), docs[2].metadata)

        self.assertEqual("0.13", docs[3].content)
        self.assertEqual(Metadata({
            "id": "001",
            "name": "doc1",
            "spec": "10ml",
            "value": 1,
            "delta": 0.13,
            "__type__": "RECORD",
            "__record_field__": "delta",
        }), docs[3].metadata)

    def test_to_record(self):
        doc = Document(
            content="doc1",
            metadata=Metadata({
                "id": "001",
                "name": "doc1",
                "spec": "10ml",
                "value": 1,
                "delta": 0.13,
                "__type__": "RECORD",
                "__record_field__": "name",
            }),
        )
        record = Document.to_record(doc)
        self.assertEqual({
            "id": "001",
            "name": "doc1",
            "spec": "10ml",
            "value": 1,
            "delta": 0.13,
        }, record)

    def test_to_records(self):
        doc1 = Document(
            content="doc1",
            metadata=Metadata({
                "id": "001",
                "name": "doc1",
                "spec": "10ml",
                "value": 1,
                "delta": 0.13,
                "__type__": "RECORD",
                "__record_field__": "name",
            }),
        )
        doc2 = Document(
            content="10ml",
            metadata=Metadata({
                "id": "001",
                "name": "doc1",
                "spec": "10ml",
                "value": 1,
                "delta": 0.13,
                "__type__": "RECORD",
                "__record_field__": "spec",
            }),
        )
        doc3 = Document(
            content="1",
            metadata=Metadata({
                "id": "001",
                "name": "doc1",
                "spec": "10ml",
                "value": 1,
                "delta": 0.13,
                "__type__": "RECORD",
                "__record_field__": "value",
            }),
        )
        doc4 = Document(
            content="0.13",
            metadata=Metadata({
                "id": "001",
                "name": "doc1",
                "spec": "10ml",
                "value": 1,
                "delta": 0.13,
                "__type__": "RECORD",
                "__record_field__": "delta",
            }),
        )
        docs = [doc1, doc2, doc3, doc4]
        records = Document.to_records("id", docs)
        self.assertEqual(1, len(records))
        self.assertEqual({
            "id": "001",
            "name": "doc1",
            "spec": "10ml",
            "value": 1,
            "delta": 0.13,
        }, records[0])


if __name__ == '__main__':
    unittest.main()
