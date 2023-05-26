# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import unittest
import logging

import openai
from llmsdk.common import Role, Message
from llmsdk.llm import ChatGpt


class TestChatGpt(unittest.TestCase):

    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)
        openai.debug = True
        openai.log = "debug"

    def test_generate(self):
        model = ChatGpt()
        message = Message(Role.HUMAN, "Say hello to me")
        reply = model.generate([message])
        print(reply)
        self.assertIsNotNone(reply)


if __name__ == '__main__':
    unittest.main()
