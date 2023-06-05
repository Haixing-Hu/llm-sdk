# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import unittest
import logging

from llmsdk.llm import Gpt
from llmsdk.util.openai_utils import set_debug_mode


class TestGpt(unittest.TestCase):

    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)
        set_debug_mode()

    def test_generate(self):
        model = Gpt()
        reply = model.generate("Say hello to me")
        print(reply)
        self.assertIsNotNone(reply)


if __name__ == '__main__':
    unittest.main()
