# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import unittest

from llmsdk.common import Example
from llmsdk.prompt import FewShotTextPromptTemplate


class TestFewShotTextPromptTemplate(unittest.TestCase):

    def test_constructor(self):
        p1 = FewShotTextPromptTemplate()
        self.assertEqual("{instruction}", p1.instruction_template)
        self.assertEqual("\n\n", p1.instruction_suffix)
        self.assertEqual([], p1.examples)
        self.assertEqual("input: ", p1.example_input_prefix)
        self.assertEqual("\n", p1.example_input_suffix)
        self.assertEqual("output: ", p1.example_output_prefix)
        self.assertEqual("\n\n", p1.example_output_suffix)
        self.assertEqual("{prompt}", p1.prompt_template)

        p2 = FewShotTextPromptTemplate("Translate the following text into {language}.")
        self.assertEqual("Translate the following text into {language}.",
                         p2.instruction_template)
        self.assertEqual("\n\n", p2.instruction_suffix)
        self.assertEqual([], p2.examples)
        self.assertEqual("input: ", p2.example_input_prefix)
        self.assertEqual("\n", p2.example_input_suffix)
        self.assertEqual("output: ", p2.example_output_prefix)
        self.assertEqual("\n\n", p2.example_output_suffix)
        self.assertEqual("{prompt}", p2.prompt_template)

        e3 = [
            Example(input="Hello, world!", output="你好，世界！"),
            Example(input="What's your name?", output="你叫什么名字？"),
        ]
        p3 = FewShotTextPromptTemplate(
            "Translate the following text into {language}.",
            examples=e3,
        )
        self.assertEqual("Translate the following text into {language}.",
                         p3.instruction_template)
        self.assertEqual("\n\n", p3.instruction_suffix)
        self.assertEqual(e3, p3.examples)
        self.assertEqual("input: ", p3.example_input_prefix)
        self.assertEqual("\n", p3.example_input_suffix)
        self.assertEqual("output: ", p3.example_output_prefix)
        self.assertEqual("\n\n", p3.example_output_suffix)
        self.assertEqual("{prompt}", p3.prompt_template)

    def test_format(self):
        i1 = "Translate the following text into {language}."
        p1 = FewShotTextPromptTemplate(instruction_template=i1)
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

        p2 = FewShotTextPromptTemplate()
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


if __name__ == '__main__':
    unittest.main()
