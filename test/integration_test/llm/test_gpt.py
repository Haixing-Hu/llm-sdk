# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import unittest
import logging

import openai
from llmsdk.llm import Gpt


class TestGpt(unittest.TestCase):

    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)
        openai.debug = True
        openai.log = "debug"

    def test_generate(self):
        model = Gpt()
        message = model.generate("Say hello to me")
        print(message)
        self.assertIsNotNone(message)


if __name__ == '__main__':
    unittest.main()