# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import unittest

from llmsdk.common import Example
from llmsdk.prompt import TextPromptTemplate


class TestTextPromptTemplate(unittest.TestCase):

    def test_constructor(self):
        p1 = TextPromptTemplate()
        self.assertEqual("", p1.instruction_template)
        self.assertEqual("\n\n", p1.instruction_suffix)
        self.assertEqual([], p1.examples)
        self.assertEqual("input: ", p1.example_input_prefix)
        self.assertEqual("\n", p1.example_input_suffix)
        self.assertEqual("output: ", p1.example_output_prefix)
        self.assertEqual("\n\n", p1.example_output_suffix)
        self.assertEqual("", p1.prompt_template)

        p2 = TextPromptTemplate("Translate the following text into {language}.")
        self.assertEqual("", p2.instruction_template)
        self.assertEqual("\n\n", p2.instruction_suffix)
        self.assertEqual([], p2.examples)
        self.assertEqual("input: ", p2.example_input_prefix)
        self.assertEqual("\n", p2.example_input_suffix)
        self.assertEqual("output: ", p2.example_output_prefix)
        self.assertEqual("\n\n", p2.example_output_suffix)
        self.assertEqual("Translate the following text into {language}.",
                         p2.prompt_template)

        e3 = [
            Example(input="Hello, world!", output="你好，世界！"),
            Example(input="What's your name?", output="你叫什么名字？"),
        ]
        p3 = TextPromptTemplate(
            "Translate the following text into {language}.",
            examples=e3,
        )
        self.assertEqual("", p3.instruction_template)
        self.assertEqual("\n\n", p3.instruction_suffix)
        self.assertEqual(e3, p3.examples)
        self.assertEqual("input: ", p3.example_input_prefix)
        self.assertEqual("\n", p3.example_input_suffix)
        self.assertEqual("output: ", p3.example_output_prefix)
        self.assertEqual("\n\n", p3.example_output_suffix)
        self.assertEqual("Translate the following text into {language}.",
                         p3.prompt_template)

    def test_format(self):
        p1 = TextPromptTemplate(
            instruction_template="Translate the following text into {language}.",
            prompt_template="{prompt}"
        )
        p1.examples.append(Example(input="Hello, world!", output="你好，世界！"))
        p1.examples.append(Example(input="What's your name?", output="你叫什么名字？"))
        v1 = p1.format(language="Chinese", prompt="Today is Sunday.")
        print(f"v1={v1}")
        self.assertEqual("Translate the following text into Chinese.\n\n"
                         "input: Hello, world!\n"
                         "output: 你好，世界！\n\n"                     
                         "input: What's your name?\n"
                         "output: 你叫什么名字？\n\n"
                         "input: Today is Sunday.\n"
                         "output: ", v1)

        p2 = TextPromptTemplate(
            instruction_template="{instruction}",
            prompt_template="{prompt}"
        )
        p2.examples.append(Example(input="Hello, world!", output="你好，世界！"))
        p2.examples.append(Example(input="What's your name?", output="你叫什么名字？"))
        v2 = p2.format(instruction="Translate the following text into Chinese.",
                       prompt="Today is Sunday.")
        print(f"v2={v2}")
        self.assertEqual("Translate the following text into Chinese.\n\n"
                         "input: Hello, world!\n"
                         "output: 你好，世界！\n\n"                     
                         "input: What's your name?\n"
                         "output: 你叫什么名字？\n\n"
                         "input: Today is Sunday.\n"
                         "output: ", v2)

        p3 = TextPromptTemplate("{prompt}")
        p3.examples.append(Example(input="Hello, world!", output="你好，世界！"))
        p3.examples.append(Example(input="What's your name?", output="你叫什么名字？"))
        v3 = p3.format(instruction="Translate the following text into Chinese.",
                       prompt="Today is Sunday.")
        print(f"v3={v3}")
        self.assertEqual("input: Hello, world!\n"
                         "output: 你好，世界！\n\n"                     
                         "input: What's your name?\n"
                         "output: 你叫什么名字？\n\n"
                         "input: Today is Sunday.\n"
                         "output: ", v3)

        p4 = TextPromptTemplate("{prompt}")
        v4 = p4.format(prompt="你好！")
        print(f"v4={v4}")
        self.assertEqual("你好！", v4)

        p5 = TextPromptTemplate("This is a sample prompt {f1} and {f2} and {f3}.")
        v5 = p5.format(f1="v1", f2="v2", f3="v3")
        print(f"v5={v5}")
        self.assertEqual("This is a sample prompt v1 and v2 and v3.", v5)

        p6 = TextPromptTemplate("This is a sample prompt {f1} and {f2} and {f3}.")
        with self.assertRaises(KeyError):
            p6.format(f1="v1", f2="v2")

        p7 = TextPromptTemplate(
            instruction_template="This is a sample prompt {f1} and {f2} and {f3}.")
        v7 = p7.format(f1="v1", f2="v2", f3="v3", f4="v4")
        print(f"v7={v7}")
        self.assertEqual("This is a sample prompt v1 and v2 and v3.", v7)


if __name__ == '__main__':
    unittest.main()
