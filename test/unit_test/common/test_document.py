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


if __name__ == '__main__':
    unittest.main()
