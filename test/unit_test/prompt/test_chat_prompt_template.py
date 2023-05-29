# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import unittest

from llmsdk.common import Example, Role, Message
from llmsdk.prompt import ChatPromptTemplate


class TestChatPromptTemplate(unittest.TestCase):

    def test_constructor(self):
        p1 = ChatPromptTemplate()
        self.assertEqual("", p1.instruction_template)
        self.assertEqual([], p1.examples)
        self.assertEqual("", p1.prompt_template)

        p2 = ChatPromptTemplate("Translate the following text into {language}.")
        self.assertEqual("", p2.instruction_template)
        self.assertEqual([], p2.examples)
        self.assertEqual("Translate the following text into {language}.",
                         p2.prompt_template)

        e3 = [
            Example(input="Hello, world!", output="你好，世界！"),
            Example(input="What's your name?", output="你叫什么名字？"),
        ]
        p3 = ChatPromptTemplate(
            "Translate the following text into {language}.",
            examples=e3,
        )
        self.assertEqual("", p3.instruction_template)
        self.assertEqual(e3, p3.examples)
        self.assertEqual("Translate the following text into {language}.",
                         p3.prompt_template)

    def test_format(self):
        p1 = ChatPromptTemplate(
            instruction_template="Translate the following text into {language}.",
            prompt_template="{prompt}")
        p1.examples.append(Example(input="Hello, world!", output="你好，世界！"))
        p1.examples.append(Example(input="What's your name?", output="你叫什么名字？"))
        v1 = p1.format(language="Chinese", prompt="Today is Sunday.")
        print(f"v1={v1}")
        self.assertEqual([
            Message(Role.SYSTEM, "Translate the following text into Chinese."),
            Message(Role.HUMAN, "Hello, world!"),
            Message(Role.AI, "你好，世界！"),
            Message(Role.HUMAN, "What's your name?"),
            Message(Role.AI, "你叫什么名字？"),
            Message(Role.HUMAN, "Today is Sunday."),
        ], v1)

        p2 = ChatPromptTemplate(
            instruction_template="{instruction}",
            prompt_template="{prompt}"
        )
        p2.examples.append(Example(input="Hello, world!", output="你好，世界！"))
        p2.examples.append(Example(input="What's your name?", output="你叫什么名字？"))
        v2 = p2.format(instruction="Translate the following text into Chinese.",
                       prompt="Today is Sunday.")
        print(f"v2={v2}")
        self.assertEqual([
            Message(Role.SYSTEM, "Translate the following text into Chinese."),
            Message(Role.HUMAN, "Hello, world!"),
            Message(Role.AI, "你好，世界！"),
            Message(Role.HUMAN, "What's your name?"),
            Message(Role.AI, "你叫什么名字？"),
            Message(Role.HUMAN, "Today is Sunday."),
        ], v2)

        p3 = ChatPromptTemplate("{prompt}")
        p3.examples.append(Example(input="Hello, world!", output="你好，世界！"))
        p3.examples.append(Example(input="What's your name?", output="你叫什么名字？"))
        v3 = p3.format(instruction="Translate the following text into Chinese.",
                       prompt="Today is Sunday.")
        print(f"v3={v3}")
        self.assertEqual([
            Message(Role.HUMAN, "Hello, world!"),
            Message(Role.AI, "你好，世界！"),
            Message(Role.HUMAN, "What's your name?"),
            Message(Role.AI, "你叫什么名字？"),
            Message(Role.HUMAN, "Today is Sunday."),
        ], v3)


if __name__ == '__main__':
    unittest.main()
