# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
import unittest

from llmsdk.splitter import RecursiveCharacterTextSplitter


class TestCharacterTextSplitter(unittest.TestCase):

    def test_split_texts(self):
        """Test recursive text splitter."""
        text = """Hi.\n\nI'm Harrison.\n\nHow? Are? You?\nOkay then f f f f.
This is a weird text to write, but gotta test the splittingggg some how.

Bye!\n\n-H."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=10,
                                                  chunk_overlap=1)
        output = splitter.split_text(text)
        expected_output = [
            "Hi.",
            "I'm",
            "Harrison.",
            "How? Are?",
            "You?",
            "Okay then",
            "f f f f.",
            "This is a",
            "a weird",
            "text to",
            "write, but",
            "gotta test",
            "the",
            "splittingg",
            "ggg",
            "some how.",
            "Bye!\n\n-H.",
        ]
        self.assertEqual(expected_output, output)


if __name__ == '__main__':
    unittest.main()
