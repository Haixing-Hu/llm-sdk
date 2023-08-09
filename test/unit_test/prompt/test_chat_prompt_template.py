# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
import unittest
from unittest.mock import patch, mock_open
from typing import Dict, Any
import json

from llmsdk.common import Example, Role, Message
from llmsdk.prompt import (
    ChatPromptTemplate,
    DEFAULT_INPUT_TEMPLATE,
    DEFAULT_INSTRUCTION_TEMPLATE,
    DEFAULT_CONTEXT_TEMPLATE,
    DEFAULT_OUTPUT_REQUIREMENT_TEMPLATE,
    DEFAULT_INSTRUCTION_PREFIX,
    DEFAULT_INSTRUCTION_SUFFIX,
    DEFAULT_CONTEXT_PREFIX,
    DEFAULT_CONTEXT_SUFFIX,
    DEFAULT_OUTPUT_REQUIREMENT_PREFIX,
    DEFAULT_OUTPUT_REQUIREMENT_SUFFIX,
)

TEST_CONFIGURATIONS = [{
    "instruction_template": "Template instruction 0",
    "context_template": "Template context 0",
    "output_requirement_template": "Template output indicator 0",
    "input_template": "Template input 0",
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
    "instruction_template": "Template instruction 1",
    "context_template": "Template context 1",
    "output_requirement_template": "Template output indicator 1",
    "input_template": "Template input 1",
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
    "instruction_prefix": "Instruction 1: <br/>",
    "instruction_suffix": "<br/>",
    "context_prefix": "Context 1: <br/>",
    "context_suffix": "<br/>",
    "output_requirement_prefix": "Output requirement 1:<br/>",
    "output_requirement_suffix": "<br/>",
    "example_list_prefix": "<ul>",
    "example_list_suffix": "</ul>",
    "example_input_prefix": "<li>Input: ",
    "example_input_suffix": "</li>",
    "example_output_prefix": "<li>Output: ",
    "example_output_suffix": "</li>",
}, {
    "instruction_template": "Template instruction 2",
    "context_template": "Template context 2",
    "output_requirement_template": "Template output indicator 2",
    "input_template": "Template input 2",
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
    "instruction_prefix": "Instruction 2: <br/>",
    "instruction_suffix": "<br/>",
    "context_prefix": "Context 2: <br/>",
    "context_suffix": "<br/>",
    "output_requirement_prefix": "Output requirement 2:<br/>",
    "output_requirement_suffix": "<br/>",
    "example_list_prefix": "<ul>",
    "example_list_suffix": "</ul>",
    "example_input_prefix": "<li>Input: ",
    "example_input_suffix": "</li>",
    "example_output_prefix": "<li>Output: ",
    "example_output_suffix": "</li>",
}, {
    "instruction_template": "Template instruction 3",
    "input_template": "Template input 3",
    "example_input_prefix": "question: ",
    "example_output_prefix": "answer: ",
}, {
    # empty
}]


class TestChatPromptTemplate(unittest.TestCase):

    def test_constructor(self):
        p1 = ChatPromptTemplate()
        self.assertEqual(DEFAULT_INSTRUCTION_TEMPLATE,
                         p1.instruction_template)
        self.assertEqual(DEFAULT_CONTEXT_TEMPLATE,
                         p1.context_template)
        self.assertEqual(DEFAULT_OUTPUT_REQUIREMENT_TEMPLATE,
                         p1.output_requirement_template)
        self.assertEqual(DEFAULT_INPUT_TEMPLATE,
                         p1.input_template)
        self.assertEqual(DEFAULT_INSTRUCTION_PREFIX,
                         p1.instruction_prefix)
        self.assertEqual(DEFAULT_INSTRUCTION_SUFFIX,
                         p1.instruction_suffix)
        self.assertEqual(DEFAULT_CONTEXT_PREFIX,
                         p1.context_prefix)
        self.assertEqual(DEFAULT_CONTEXT_SUFFIX,
                         p1.context_suffix)
        self.assertEqual(DEFAULT_OUTPUT_REQUIREMENT_PREFIX,
                         p1.output_requirement_prefix)
        self.assertEqual(DEFAULT_OUTPUT_REQUIREMENT_SUFFIX,
                         p1.output_requirement_suffix)
        self.assertEqual([], p1.examples)
        self.assertEqual([], p1.histories)

        p2 = ChatPromptTemplate("Translate the following text into {language}.")
        self.assertEqual("Translate the following text into {language}.",
                         p2.instruction_template)
        self.assertEqual(DEFAULT_CONTEXT_TEMPLATE,
                         p2.context_template)
        self.assertEqual(DEFAULT_OUTPUT_REQUIREMENT_TEMPLATE,
                         p2.output_requirement_template)
        self.assertEqual(DEFAULT_INPUT_TEMPLATE,
                         p2.input_template)
        self.assertEqual(DEFAULT_INSTRUCTION_PREFIX,
                         p2.instruction_prefix)
        self.assertEqual(DEFAULT_INSTRUCTION_SUFFIX,
                         p2.instruction_suffix)
        self.assertEqual(DEFAULT_CONTEXT_PREFIX,
                         p2.context_prefix)
        self.assertEqual(DEFAULT_CONTEXT_SUFFIX,
                         p2.context_suffix)
        self.assertEqual(DEFAULT_OUTPUT_REQUIREMENT_PREFIX,
                         p2.output_requirement_prefix)
        self.assertEqual(DEFAULT_OUTPUT_REQUIREMENT_SUFFIX,
                         p2.output_requirement_suffix)
        self.assertEqual([], p2.examples)
        self.assertEqual([], p2.histories)

        e3 = [
            Example(input="Hello, world!", output="你好，世界！"),
            Example(input="What's your name?", output="你叫什么名字？"),
        ]
        p3 = ChatPromptTemplate(
            "Translate the following text into {language}.",
            examples=e3,
        )
        self.assertEqual("Translate the following text into {language}.",
                         p3.instruction_template)
        self.assertEqual(DEFAULT_CONTEXT_TEMPLATE,
                         p3.context_template)
        self.assertEqual(DEFAULT_OUTPUT_REQUIREMENT_TEMPLATE,
                         p3.output_requirement_template)
        self.assertEqual(DEFAULT_INPUT_TEMPLATE,
                         p3.input_template)
        self.assertEqual(DEFAULT_INSTRUCTION_PREFIX,
                         p3.instruction_prefix)
        self.assertEqual(DEFAULT_INSTRUCTION_SUFFIX,
                         p3.instruction_suffix)
        self.assertEqual(DEFAULT_CONTEXT_PREFIX,
                         p3.context_prefix)
        self.assertEqual(DEFAULT_CONTEXT_SUFFIX,
                         p3.context_suffix)
        self.assertEqual(DEFAULT_OUTPUT_REQUIREMENT_PREFIX,
                         p3.output_requirement_prefix)
        self.assertEqual(DEFAULT_OUTPUT_REQUIREMENT_SUFFIX,
                         p3.output_requirement_suffix)
        self.assertEqual(e3, p3.examples)
        self.assertEqual([], p3.histories)

    def test_format(self):
        p1 = ChatPromptTemplate(
            instruction_template="Translate the following text into {language}.",
            input_template="{input}")
        p1.add_example(input="Hello, world!", output="你好，世界！")
        p1.add_example(input="What's your name?", output="你叫什么名字？")
        v1 = p1.format(language="Chinese", input="Today is Sunday.")
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
            input_template="{input}"
        )
        p2.add_example(input="Hello, world!", output="你好，世界！")
        p2.add_example(input="What's your name?", output="你叫什么名字？")
        v2 = p2.format(instruction="Translate the following text into Chinese.",
                       input="Today is Sunday.")
        print(f"v2={v2}")
        self.assertEqual([
            Message(Role.SYSTEM, "Translate the following text into Chinese."),
            Message(Role.HUMAN, "Hello, world!"),
            Message(Role.AI, "你好，世界！"),
            Message(Role.HUMAN, "What's your name?"),
            Message(Role.AI, "你叫什么名字？"),
            Message(Role.HUMAN, "Today is Sunday."),
        ], v2)

        p3 = ChatPromptTemplate(instruction_template="")
        p3.add_example(input="Hello, world!", output="你好，世界！")
        p3.add_example(input="What's your name?", output="你叫什么名字？")
        v3 = p3.format(instruction="Translate the following text into Chinese.",
                       input="Today is Sunday.")
        print(f"v3={v3}")
        self.assertEqual([
            Message(Role.HUMAN, "Hello, world!"),
            Message(Role.AI, "你好，世界！"),
            Message(Role.HUMAN, "What's your name?"),
            Message(Role.AI, "你叫什么名字？"),
            Message(Role.HUMAN, "Today is Sunday."),
        ], v3)

        p4 = ChatPromptTemplate()
        p4.add_history(human_message="Who won the world series in 2020?",
                       ai_message="The Los Angeles Dodgers won the World Series in 2020.")
        v4 = p4.format(instruction="You are a helpful assistant.",
                       input="Where was it played?")
        print(f"v4={v4}")
        self.assertEqual([
            Message(Role.SYSTEM, "You are a helpful assistant."),
            Message(Role.HUMAN, "Who won the world series in 2020?"),
            Message(Role.AI, "The Los Angeles Dodgers won the World Series in 2020."),
            Message(Role.HUMAN, "Where was it played?"),
        ], v4)

    def test_format_with_history(self):
        p8 = ChatPromptTemplate()
        p8.add_history(human_message="Who won the world series in 2020?",
                       ai_message="The Los Angeles Dodgers won the World Series in 2020.")
        v8 = p8.format(instruction="You are a helpful assistant.",
                       input="Where was it played?")
        self.assertEqual([
            Message(Role.SYSTEM, "You are a helpful assistant."),
            Message(Role.HUMAN, "Who won the world series in 2020?"),
            Message(Role.AI, "The Los Angeles Dodgers won the World Series in 2020."),
            Message(Role.HUMAN, "Where was it played?"),
        ], v8)

    def test_format_with_context(self):
        p8 = ChatPromptTemplate()
        p8.context_template = "Context: {context}"
        p8.add_history(human_message="Who won the world series in 2020?",
                       ai_message="The Los Angeles Dodgers won the World Series in 2020.")
        v8 = p8.format(instruction="You are a helpful assistant.",
                       input="Where was it played?",
                       context="In the World Series 2020 in Arlington, Texas， "
                               "Los Angeles Dodgers beat Tampa Bay Rays 4-2 and "
                               "won the first championship in 32 years.")
        self.assertEqual([
            Message(Role.SYSTEM, "You are a helpful assistant.\n\n"
                                 "The following are known context:\n"
                                 "Context: In the World Series 2020 in Arlington, Texas， "
                                 "Los Angeles Dodgers beat Tampa Bay Rays 4-2 and "
                                 "won the first championship in 32 years."),
            Message(Role.HUMAN, "Who won the world series in 2020?"),
            Message(Role.AI, "The Los Angeles Dodgers won the World Series in 2020."),
            Message(Role.HUMAN, "Where was it played?"),
        ], v8)

    def test_format_with_context_and_output_requirement(self):
        p8 = ChatPromptTemplate()
        p8.context_template = "Context: {context}"
        p8.add_history(human_message="Who won the world series in 2020?",
                       ai_message="{'answer': 'Los Angeles Dodgers'}")
        v8 = p8.format(instruction="You are a helpful assistant.",
                       input="Where was it played?",
                       context="In the World Series 2020 in Arlington, Texas， "
                               "Los Angeles Dodgers beat Tampa Bay Rays 4-2 and "
                               "won the first championship in 32 years.",
                       output_requirement="The output must be a JSON object.")
        self.assertEqual([
            Message(Role.SYSTEM, "You are a helpful assistant.\n\n"
                                 "The following are known context:\n"
                                 "Context: In the World Series 2020 in Arlington, Texas， "
                                 "Los Angeles Dodgers beat Tampa Bay Rays 4-2 and "
                                 "won the first championship in 32 years.\n\n"
                                 "The output must satisfy the following requirements:\n"
                                 "The output must be a JSON object."),
            Message(Role.HUMAN, "Who won the world series in 2020?"),
            Message(Role.AI, "{'answer': 'Los Angeles Dodgers'}"),
            Message(Role.HUMAN, "Where was it played?"),
        ], v8)

    def _check_load_result(self,
                           template: ChatPromptTemplate,
                           conf: Dict[str, Any]):
        self.assertEqual(conf.get("instruction_template",
                                  DEFAULT_INSTRUCTION_TEMPLATE),
                         template.instruction_template)
        self.assertEqual(conf.get("context_template",
                                  DEFAULT_CONTEXT_TEMPLATE),
                         template.context_template)
        self.assertEqual(conf.get("output_requirement_template",
                                  DEFAULT_OUTPUT_REQUIREMENT_TEMPLATE),
                         template.output_requirement_template)
        self.assertEqual(conf.get("input_template",
                                  DEFAULT_INPUT_TEMPLATE),
                         template.input_template)
        self.assertEqual(conf.get("instruction_prefix",
                                  DEFAULT_INSTRUCTION_PREFIX),
                         template.instruction_prefix)
        self.assertEqual(conf.get("instruction_suffix",
                                  DEFAULT_INSTRUCTION_SUFFIX),
                         template.instruction_suffix)
        self.assertEqual(conf.get("context_prefix",
                                  DEFAULT_CONTEXT_PREFIX),
                         template.context_prefix)
        self.assertEqual(conf.get("context_suffix",
                                  DEFAULT_CONTEXT_SUFFIX),
                         template.context_suffix)
        self.assertEqual(conf.get("output_requirement_prefix",
                                  DEFAULT_OUTPUT_REQUIREMENT_PREFIX),
                         template.output_requirement_prefix)
        self.assertEqual(conf.get("output_requirement_suffix",
                                  DEFAULT_OUTPUT_REQUIREMENT_SUFFIX),
                         template.output_requirement_suffix)
        if "examples" in conf:
            self.assertEqual(len(conf["examples"]), len(template.examples))
            for c_example, t_example in zip(conf["examples"], template.examples):
                self.assertEqual(c_example.get("id", None), t_example.id)
                self.assertEqual(c_example.get("input"), t_example.input)
                self.assertEqual(c_example.get("output"), t_example.output)
        if "histories" in conf:
            self.assertEqual(len(conf["histories"]), len(template.histories))
            for c_history, t_history in zip(conf["histories"], template.histories):
                self.assertEqual(c_history.get("role"), t_history.role.value)
                self.assertEqual(c_history.get("content"), t_history.content)
                self.assertEqual(c_history.get("name", None), t_history.name)

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
