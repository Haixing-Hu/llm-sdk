# ==============================================================================
#                                                                              =
#      Copyright (c) 2023. Haixing Hu                                          =
#      All rights reserved.                                                    =
#                                                                              =
# ==============================================================================
import unittest

from llmsdk.splitter import CharacterTextSplitter


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


if __name__ == '__main__':
    unittest.main()
