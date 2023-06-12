# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import unittest
from unittest.mock import patch, mock_open
from typing import Dict, Any
import json

from llmsdk.common import Example
from llmsdk.prompt import (
    TextPromptTemplate,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_INSTRUCTION_TEMPLATE,
    DEFAULT_INSTRUCTION_SUFFIX,
    DEFAULT_EXAMPLE_LIST_PREFIX,
    DEFAULT_EXAMPLE_INPUT_PREFIX,
    DEFAULT_EXAMPLE_INPUT_SUFFIX,
    DEFAULT_EXAMPLE_OUTPUT_PREFIX,
    DEFAULT_EXAMPLE_OUTPUT_SUFFIX,
)


TEST_CONFIGURATIONS = [{
    "instruction_template": "Template instruction",
    "prompt_template": "Template prompt",
    "examples": [
        {
            "id": "1",
            "input": "Input 1",
            "output": "Output 1"
        },
        {
            "id": "2",
            "input": "Input 2",
            "output": "Output 2"
        },
        {
            "input": "Input 3",
            "output": "Output 3"
        }
    ],
    "instruction_suffix": "\n",
}, {
    "instruction_template": "Template instruction",
    "prompt_template": "Template prompt",
    "examples": [
        {
            "id": "1",
            "input": "Input 1",
            "output": "Output 1"
        },
        {
            "id": "2",
            "input": "Input 2",
            "output": "Output 2"
        }
    ],
    "instruction_suffix": "<br/>",
    "example_list_prefix": "<ul>",
    "example_input_prefix": "<li>Input: ",
    "example_input_suffix": "</li>",
    "example_output_prefix": "<li>Output: ",
    "example_output_suffix": "</li></ul>",
}, {
    "instruction_template": "Template instruction",
    "prompt_template": "Template prompt",
    "examples": [
        {
            "id": "1",
            "input": "Input 1",
            "output": "Output 1"
        },
        {
            "id": "2",
            "input": "Input 2",
            "output": "Output 2"
        }
    ],
    "histories": [
        {
            "role": "Human",
            "content": "Input 3"
        },
        {
            "role": "AI",
            "content": "Output 3"
        }
    ],
    "instruction_suffix": "<br/>",
    "example_list_prefix": "<ul>",
    "example_input_prefix": "<li>Input: ",
    "example_input_suffix": "</li>",
    "example_output_prefix": "<li>Output: ",
    "example_output_suffix": "</li></ul>",
}, {
    "instruction_template": "Template instruction",
    "prompt_template": "Template prompt",
    "example_input_prefix": "question: ",
    "example_output_prefix": "answer: ",
}, {
    # empty
}]


class TestTextPromptTemplate(unittest.TestCase):

    def test_constructor(self):
        p1 = TextPromptTemplate()
        self.assertEqual(DEFAULT_INSTRUCTION_TEMPLATE,
                         p1.instruction_template)
        self.assertEqual(DEFAULT_PROMPT_TEMPLATE,
                         p1.prompt_template)
        self.assertEqual(DEFAULT_INSTRUCTION_SUFFIX,
                         p1.instruction_suffix)
        self.assertEqual(DEFAULT_EXAMPLE_LIST_PREFIX,
                         p1.example_list_prefix)
        self.assertEqual(DEFAULT_EXAMPLE_INPUT_PREFIX,
                         p1.example_input_prefix)
        self.assertEqual(DEFAULT_EXAMPLE_INPUT_SUFFIX,
                         p1.example_input_suffix)
        self.assertEqual(DEFAULT_EXAMPLE_OUTPUT_PREFIX,
                         p1.example_output_prefix)
        self.assertEqual(DEFAULT_EXAMPLE_OUTPUT_SUFFIX,
                         p1.example_output_suffix)
        self.assertEqual([], p1.examples)

        p2 = TextPromptTemplate("Translate the following text into {language}.")
        self.assertEqual(DEFAULT_INSTRUCTION_TEMPLATE,
                         p2.instruction_template)
        self.assertEqual("Translate the following text into {language}.",
                         p2.prompt_template)
        self.assertEqual(DEFAULT_INSTRUCTION_SUFFIX,
                         p2.instruction_suffix)
        self.assertEqual(DEFAULT_EXAMPLE_LIST_PREFIX,
                         p2.example_list_prefix)
        self.assertEqual(DEFAULT_EXAMPLE_INPUT_PREFIX,
                         p2.example_input_prefix)
        self.assertEqual(DEFAULT_EXAMPLE_INPUT_SUFFIX,
                         p2.example_input_suffix)
        self.assertEqual(DEFAULT_EXAMPLE_OUTPUT_PREFIX,
                         p2.example_output_prefix)
        self.assertEqual(DEFAULT_EXAMPLE_OUTPUT_SUFFIX,
                         p2.example_output_suffix)
        self.assertEqual([], p2.examples)

        e3 = [
            Example(input="Hello, world!", output="你好，世界！"),
            Example(input="What's your name?", output="你叫什么名字？"),
        ]
        p3 = TextPromptTemplate(
            "Translate the following text into {language}.",
            examples=e3,
        )
        self.assertEqual(DEFAULT_INSTRUCTION_TEMPLATE,
                         p3.instruction_template)
        self.assertEqual("Translate the following text into {language}.",
                         p3.prompt_template)
        self.assertEqual(DEFAULT_INSTRUCTION_SUFFIX,
                         p3.instruction_suffix)
        self.assertEqual(DEFAULT_EXAMPLE_LIST_PREFIX,
                         p3.example_list_prefix)
        self.assertEqual(DEFAULT_EXAMPLE_INPUT_PREFIX,
                         p3.example_input_prefix)
        self.assertEqual(DEFAULT_EXAMPLE_INPUT_SUFFIX,
                         p3.example_input_suffix)
        self.assertEqual(DEFAULT_EXAMPLE_OUTPUT_PREFIX,
                         p3.example_output_prefix)
        self.assertEqual(DEFAULT_EXAMPLE_OUTPUT_SUFFIX,
                         p3.example_output_suffix)
        self.assertEqual(e3, p3.examples)

    def test_format(self):
        p1 = TextPromptTemplate(
            instruction_template="Translate the following text into {language}.",
            prompt_template="{prompt}"
        )
        p1.add_example(input="Hello, world!", output="你好，世界！")
        p1.add_example(input="What's your name?", output="你叫什么名字？")
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
        p2.add_example(input="Hello, world!", output="你好，世界！")
        p2.add_example(input="What's your name?", output="你叫什么名字？")
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

        p3 = TextPromptTemplate(instruction_template="")
        p3.add_example(input="Hello, world!", output="你好，世界！")
        p3.add_example(input="What's your name?", output="你叫什么名字？")
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

    def test_format_with_history(self):
        p8 = TextPromptTemplate()
        p8.add_history(human_message="Who won the world series in 2020?",
                       ai_message="The Los Angeles Dodgers won the World Series in 2020.")
        v8 = p8.format(instruction="You are a helpful assistant.",
                       prompt="Where was it played?")
        self.assertEqual("You are a helpful assistant.\n\n"
                         "input: Who won the world series in 2020?\n"
                         "output: The Los Angeles Dodgers won the World Series in 2020.\n\n"
                         "input: Where was it played?\n"
                         "output: ", v8)

    def _check_load_result(self,
                           template: TextPromptTemplate,
                           conf: Dict[str, Any]):
        self.assertEqual(conf.get("instruction_template",
                                  DEFAULT_INSTRUCTION_TEMPLATE),
                         template.instruction_template)
        self.assertEqual(conf.get("prompt_template",
                                  DEFAULT_PROMPT_TEMPLATE),
                         template.prompt_template)
        self.assertEqual(conf.get("instruction_suffix",
                                  DEFAULT_INSTRUCTION_SUFFIX),
                         template.instruction_suffix)
        self.assertEqual(conf.get("example_list_prefix",
                                  DEFAULT_EXAMPLE_LIST_PREFIX),
                         template.example_list_prefix)
        self.assertEqual(conf.get("example_input_prefix",
                                  DEFAULT_EXAMPLE_INPUT_PREFIX),
                         template.example_input_prefix)
        self.assertEqual(conf.get("example_input_suffix",
                                  DEFAULT_EXAMPLE_INPUT_SUFFIX),
                         template.example_input_suffix)
        self.assertEqual(conf.get("example_output_prefix",
                                  DEFAULT_EXAMPLE_OUTPUT_PREFIX),
                         template.example_output_prefix)
        self.assertEqual(conf.get("example_output_suffix",
                                  DEFAULT_EXAMPLE_OUTPUT_SUFFIX),
                         template.example_output_suffix)
        if "examples" in conf:
            self.assertEqual(len(conf["examples"]),
                             len(template.examples))
            for c_example, t_example in zip(conf["examples"], template.examples):
                self.assertEqual(c_example.get("id", None), t_example.id)
                self.assertEqual(c_example.get("input"), t_example.input)
                self.assertEqual(c_example.get("output"), t_example.output)

    def _test_load_from_file(self,
                             template: TextPromptTemplate,
                             conf: Dict[str, Any]):
        text = json.dumps(conf)
        with patch('builtins.open', mock_open(read_data=text)):
            template.load_from_file('config.json')
        self._check_load_result(template, conf)

    def test_load(self):
        template = TextPromptTemplate()
        for conf in TEST_CONFIGURATIONS:
            template.load(conf)
            self._check_load_result(template, conf)

    def test_load_from_file(self):
        template = TextPromptTemplate()
        for conf in TEST_CONFIGURATIONS:
            self._test_load_from_file(template, conf)


if __name__ == '__main__':
    unittest.main()
