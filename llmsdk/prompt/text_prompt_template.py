# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from dataclasses import dataclass
from typing import Any, Dict, List

from ..common.example import Example
from ..common.message import Message
from .structured_prompt_template import StructuredPromptTemplate

DEFAULT_EXAMPLE_LIST_PREFIX = ""
"""
The default prefix for the list of examples.
"""

DEFAULT_EXAMPLE_LIST_SUFFIX = ""
"""
The default suffix for the list of examples.
"""

DEFAULT_EXAMPLE_INPUT_PREFIX: str = "input: "
"""
The default prefix for the input of an example.
"""

DEFAULT_EXAMPLE_INPUT_SUFFIX: str = "\n"
"""
The default suffix for the input of an example.
"""

DEFAULT_EXAMPLE_OUTPUT_PREFIX: str = "output: "
"""
The default prefix for the output of an example.
"""

DEFAULT_EXAMPLE_OUTPUT_SUFFIX: str = "\n\n"
"""
The default suffix for the input of an example.
"""


@dataclass
class TextPromptTemplate(StructuredPromptTemplate):
    """
    The prompt template used to format the few-shot prompts in the
    text-completion models.

    The formatted text prompt has the following form:

    ```
    {instruction_prefix}{formatted_instruction}{instruction_suffix}
    {context_prefix}{formatted_context}{context_suffix}
    {output_requirement_prefix}{formatted_output_requirement}{output_requirement_suffix}
    {example_input_prefix}{examples[0].input}{example_input_suffix}
    {example_output_prefix}{examples[0].output}{example_output_suffix}
    {example_input_prefix}{examples[1].input}{example_input_suffix}
    {example_output_prefix}{examples[1].output}{example_output_suffix}
    {example_input_prefix}{examples[2].input}{example_input_suffix}
    {example_output_prefix}{examples[2].output}{example_output_suffix}
    ...
    {example_input_prefix}{histories[0].content}{example_input_suffix}
    {example_output_prefix}{histories[1].content}{example_output_suffix}
    {example_input_prefix}{histories[2].content}{example_input_suffix}
    {example_output_prefix}{histories[3].content}{example_output_suffix}
    {example_input_prefix}{histories[4].content}{example_input_suffix}
    {example_output_prefix}{histories[5].content}{example_output_suffix}
    ...
    {example_input_prefix}{formatted_input}{example_input_suffix}
    {example_output_prefix}
    ```

    Examples:

    ```
        i1 = "Translate the following text into {language}."
        p1 = TextPromptTemplate(instruction_template=i1)
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

        p2 = TextPromptTemplate()
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

        p3 = ChatPromptTemplate("You are a helpful assistant.")
        p3.add_history(human_message="Who won the world series in 2020?",
                       ai_message="The Los Angeles Dodgers won the World Series in 2020.")
        v3 = p3.format(prompt="Where was it played?")
        print(f"v3={v3}")
        self.assertEqual("You are a helpful assistant.\n\n"
                         "input: Who won the world series in 2020?\n"
                         "output: The Los Angeles Dodgers won the World Series in 2020.\n\n"
                         "input: Where was it played?\n"
                         "output: ", v3)
    ```

    """

    example_list_prefix: str = DEFAULT_EXAMPLE_LIST_PREFIX
    """
    The prefix for the list of examples.
    """

    example_list_suffix: str = DEFAULT_EXAMPLE_LIST_SUFFIX
    """
    The suffix for the list of examples.
    """

    example_input_prefix: str = DEFAULT_EXAMPLE_INPUT_PREFIX
    """
    The prefix for the input of an example.
    """

    example_input_suffix: str = DEFAULT_EXAMPLE_INPUT_SUFFIX
    """
    The suffix for the input of an example.
    """

    example_output_prefix: str = DEFAULT_EXAMPLE_OUTPUT_PREFIX
    """
    The prefix for the output of an example.
    """

    example_output_suffix: str = DEFAULT_EXAMPLE_OUTPUT_SUFFIX
    """
    The suffix for the input of an example.
    """

    def format_prompt(self, **kwargs: Any) -> str:
        instruction = self._format_instruction(**kwargs)
        context = self._format_context(**kwargs)
        output_requirement = self._format_output_requirement(**kwargs)
        examples = [self._format_example(e) for e in self.examples]
        histories = self._format_histories(self.histories)
        input = self._format_input(**kwargs)
        result = instruction + context + output_requirement
        if len(examples) > 0 or len(histories) > 0:
            result += self.example_list_prefix
            result += "".join(examples)
            result += histories
            result += self.example_list_suffix
        if len(input) > 0:
            result += self.example_input_prefix + input.strip() + self.example_input_suffix
            result += self.example_output_prefix
        return result.strip()

    def format_explanation_prompt(self, last_reply: str, **kwargs: Any) -> str:
        last_prompt = self.format_prompt(**kwargs)
        return (last_prompt + " " + last_reply + self.example_output_suffix
                + self.explanation_instruction_prefix
                + self.explanation_instruction
                + self.explanation_instruction_suffix)

    def _format_example(self, example: Example) -> str:
        """
        Formats the input/output of an example.
        """
        return (self.example_input_prefix
                + example.input.strip()
                + self.example_input_suffix
                + self.example_output_prefix
                + example.output.strip()
                + self.example_output_suffix)

    def _format_histories(self, histories: List[Message]) -> str:
        """
        Formats the conversation histories as a list of input/output pairs.
        """
        if len(histories) % 2 != 0:
            raise ValueError("The number of messages must be even.")
        result = ""
        for i in range(0, len(histories), 2):
            result += (self.example_input_prefix
                       + histories[i].content.strip()
                       + self.example_input_suffix
                       + self.example_output_prefix
                       + histories[i + 1].content.strip()
                       + self.example_output_suffix)
        return result

    def load(self, config: Dict[str, str]) -> None:
        super().load(config)
        self.example_list_prefix = config.get("example_list_prefix",
                                              DEFAULT_EXAMPLE_LIST_PREFIX)
        self.example_list_suffix = config.get("example_list_suffix",
                                              DEFAULT_EXAMPLE_LIST_SUFFIX)
        self.example_input_prefix = config.get("example_input_prefix",
                                               DEFAULT_EXAMPLE_INPUT_PREFIX)
        self.example_input_suffix = config.get("example_input_suffix",
                                               DEFAULT_EXAMPLE_INPUT_SUFFIX)
        self.example_output_prefix = config.get("example_output_prefix",
                                                DEFAULT_EXAMPLE_OUTPUT_PREFIX)
        self.example_output_suffix = config.get("example_output_suffix",
                                                DEFAULT_EXAMPLE_OUTPUT_SUFFIX)
