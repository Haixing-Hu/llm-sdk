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

from llmsdk.common import Example, Role, Message
from llmsdk.prompt import StructuredPromptTemplate, ChatPromptTemplate

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
    "example_input_prefix": "question: ",
    "example_output_prefix": "answer: ",
}, {
    # empty
}]


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

    def _check_load_result(self,
                           template: ChatPromptTemplate,
                           conf: Dict[str, Any]):
        self.assertEqual(conf.get("instruction_template",
                                  StructuredPromptTemplate.DEFAULT_INSTRUCTION_TEMPLATE),
                         template.instruction_template)
        self.assertEqual(conf.get("prompt_template",
                                  StructuredPromptTemplate.DEFAULT_PROMPT_TEMPLATE),
                         template.prompt_template)
        if "examples" in conf:
            self.assertEqual(len(conf["examples"]),
                             len(template.examples))
            for c_example, t_example in zip(conf["examples"], template.examples):
                self.assertEqual(c_example.get("id", None), t_example.id)
                self.assertEqual(c_example.get("input"), t_example.input)
                self.assertEqual(c_example.get("output"), t_example.output)

    def _test_load_from_file(self,
                             template: ChatPromptTemplate,
                             conf: Dict[str, Any]):
        text = json.dumps(conf)
        with patch('builtins.open', mock_open(read_data=text)):
            template.load_from_file('config.json')
        self._check_load_result(template, conf)

    def test_load(self):
        template = ChatPromptTemplate()
        for conf in TEST_CONFIGURATIONS:
            template.load(conf)
            self._check_load_result(template, conf)

    def test_load_from_file(self):
        template = ChatPromptTemplate()
        for conf in TEST_CONFIGURATIONS:
            self._test_load_from_file(template, conf)


if __name__ == '__main__':
    unittest.main()
