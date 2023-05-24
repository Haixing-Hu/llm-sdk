# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from dataclasses import asdict
import unittest

from llmsdk.common import ChatMessage


class TestChatMessage(unittest.TestCase):

    def test_constructor(self):
        msg1 = ChatMessage("system", "hello world")
        self.assertEqual("system", msg1.role)
        self.assertEqual("hello world", msg1.content)

    def test_asdict(self):
        msg1 = ChatMessage("system", "hello world")
        d = asdict(msg1)
        print(d)
        self.assertEqual({"role": "system", "content": "hello world"}, d)


if __name__ == '__main__':
    unittest.main()
