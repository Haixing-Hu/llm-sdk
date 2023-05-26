# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import unittest

from llmsdk.common import Role, Message


class TestChatMessage(unittest.TestCase):

    def test_constructor(self):
        msg1 = Message(Role.SYSTEM, "hello world")
        self.assertEqual(Role.SYSTEM, msg1.role)
        self.assertEqual("hello world", msg1.content)

    def test_to_dict(self):
        msg1 = Message(Role.SYSTEM, "hello world")
        d1 = msg1.to_dict()
        self.assertEqual({"role": "System", "content": "hello world"}, d1)

        msg2 = Message(Role.HUMAN, "message 2", name="Tom")
        d2 = msg2.to_dict()
        print(d2)
        self.assertEqual({"role": "Human", "content": "message 2", "name": "Tom"}, d2)

        msg3 = Message(Role.AI, "message 3")
        d3 = msg3.to_dict()
        print(d3)
        self.assertEqual({"role": "AI", "content": "message 3"}, d3)

    def test_to_dict_with_role_names_map(self):
        role_names_map = {
            Role.SYSTEM: "system",
            Role.HUMAN: "user",
            Role.AI: "assistant"
        }
        msg1 = Message(Role.SYSTEM, "hello world")
        d1 = msg1.to_dict(role_names_map)
        self.assertEqual({"role": "system", "content": "hello world"}, d1)

        msg2 = Message(Role.HUMAN, "message 2", name="Tom")
        d2 = msg2.to_dict(role_names_map)
        print(d2)
        self.assertEqual({"role": "user", "content": "message 2", "name": "Tom"}, d2)

        msg3 = Message(Role.AI, "message 3")
        d3 = msg3.to_dict(role_names_map)
        print(d3)
        self.assertEqual({"role": "assistant", "content": "message 3"}, d3)


if __name__ == '__main__':
    unittest.main()
