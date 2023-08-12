# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
import unittest

from llmsdk.splitter import CharacterTextSplitter
from llmsdk.common import Document, Metadata


class TestCharacterTextSplitter(unittest.TestCase):

    def test_split_texts(self):
        """Test splitting by character count."""
        text = "foo bar baz 123"
        splitter = CharacterTextSplitter(separator=" ",
                                         chunk_size=7,
                                         chunk_overlap=3)
        output = splitter.split_text(text)
        expected_output = ["foo bar", "bar baz", "baz 123"]
        self.assertEqual(expected_output, output)

    def test_character_text_splitter_empty_doc(self):
        """Test splitting by character count doesn't create empty documents."""
        text = "foo  bar"
        splitter = CharacterTextSplitter(separator=" ",
                                         chunk_size=2,
                                         chunk_overlap=0)
        output = splitter.split_text(text)
        expected_output = ["foo", "bar"]
        self.assertEqual(expected_output, output)

    def test_character_text_splitter_separtor_empty_doc(self):
        """Test edge cases are separators."""
        text = "f b"
        splitter = CharacterTextSplitter(separator=" ",
                                         chunk_size=2,
                                         chunk_overlap=0)
        output = splitter.split_text(text)
        expected_output = ["f", "b"]
        self.assertEqual(expected_output, output)

    def test_character_text_splitter_long(self):
        """Test splitting by character count on long words."""
        text = "foo bar baz a a"
        splitter = CharacterTextSplitter(separator=" ",
                                         chunk_size=3,
                                         chunk_overlap=1)
        output = splitter.split_text(text)
        expected_output = ["foo", "bar", "baz", "a a"]
        self.assertEqual(expected_output, output)

    def test_character_text_splitter_short_words_first(self):
        """Test splitting by character count when shorter words are first."""
        text = "a a foo bar baz"
        splitter = CharacterTextSplitter(separator=" ",
                                         chunk_size=3,
                                         chunk_overlap=1)
        output = splitter.split_text(text)
        expected_output = ["a a", "foo", "bar", "baz"]
        self.assertEqual(expected_output, output)

    def test_character_text_splitter_longer_words(self):
        """Test splitting by characters when splits not found easily."""
        text = "foo bar baz 123"
        splitter = CharacterTextSplitter(separator=" ",
                                         chunk_size=1,
                                         chunk_overlap=1)
        output = splitter.split_text(text)
        expected_output = ["foo", "bar", "baz", "123"]
        self.assertEqual(expected_output, output)

    def test_character_text_splitting_args(self):
        """Test invalid arguments."""
        with self.assertRaises(ValueError):
            CharacterTextSplitter(chunk_size=2, chunk_overlap=4)

    def test_split_documents(self):
        """Test split_documents."""
        splitter = CharacterTextSplitter(separator="",
                                         chunk_size=1,
                                         chunk_overlap=0)
        docs = [
            Document(id="a", content="foo", metadata=Metadata({"source": "1"})),
            Document(id="b", content="bar", metadata=Metadata({"source": "2"})),
            Document(id="c", content="baz", metadata=Metadata({"source": "1"})),
        ]
        expected_output = [
            Document(id="a-0",
                     content="f",
                     metadata=Metadata({
                         "source": "1",
                         "__original_document_id__": "a",
                         "__original_document_index__": 0,
                     })),
            Document(id="a-1",
                     content="o",
                     metadata=Metadata({
                         "source": "1",
                         "__original_document_id__": "a",
                         "__original_document_index__": 1,
                     })),
            Document(id="a-2",
                     content="o",
                     metadata=Metadata({
                         "source": "1",
                         "__original_document_id__": "a",
                         "__original_document_index__": 2,
                     })),
            Document(id="b-0",
                     content="b",
                     metadata=Metadata({
                         "source": "2",
                         "__original_document_id__": "b",
                         "__original_document_index__": 0,
                     })),
            Document(id="b-1",
                     content="a",
                     metadata=Metadata({
                         "source": "2",
                         "__original_document_id__": "b",
                         "__original_document_index__": 1,
                     })),
            Document(id="b-2",
                     content="r",
                     metadata=Metadata({
                         "source": "2",
                         "__original_document_id__": "b",
                         "__original_document_index__": 2,
                     })),
            Document(id="c-0",
                     content="b",
                     metadata=Metadata({
                         "source": "1",
                         "__original_document_id__": "c",
                         "__original_document_index__": 0,
                     })),
            Document(id="c-1",
                     content="a",
                     metadata=Metadata({
                         "source": "1",
                         "__original_document_id__": "c",
                         "__original_document_index__": 1,
                     })),
            Document(id="c-2",
                     content="z",
                     metadata=Metadata({
                         "source": "1",
                         "__original_document_id__": "c",
                         "__original_document_index__": 2,
                     })),
        ]
        output = splitter.split_documents(docs)
        self.assertEqual(expected_output, output)


if __name__ == '__main__':
    unittest.main()
