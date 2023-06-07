# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import unittest
import logging

from llmsdk.common import Role, Message
from llmsdk.llm import ChatGpt
from llmsdk.util.openai_utils import set_openai_debug_mode


class TestChatGpt(unittest.TestCase):

    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)
        set_openai_debug_mode()

    def test_generate(self):
        model = ChatGpt()
        message = Message(Role.HUMAN, "Say hello to me")
        reply = model.generate([message])
        print(reply)
        self.assertIsNotNone(reply)


if __name__ == '__main__':
    unittest.main()
