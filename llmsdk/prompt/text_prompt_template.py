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


DEFAULT_INSTRUCTION_SUFFIX: str = "\n"
"""
The default suffix for the instruction.
"""

DEFAULT_EXAMPLE_LIST_PREFIX = "\n"
"""
The default prefix for the list of examples.
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
    The prompt template used to format the few-shot prompts.

    The formatted text prompt has the following form:

    ```
    {instruction}{instruction_suffix}
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
    {example_input_prefix}{prompt}{example_input_suffix}
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

    instruction_suffix: str = DEFAULT_INSTRUCTION_SUFFIX
    """
    The suffix for the instruction.
    """

    example_list_prefix: str = DEFAULT_EXAMPLE_LIST_PREFIX
    """
    The prefix for the list of examples.
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

    def format(self, **kwargs: Any) -> str:
        instruction = self._format_instruction(**kwargs)
        examples = [self._format_example(e) for e in self.examples]
        histories = self._format_histories(self.histories)
        prompt = self._format_prompt(**kwargs)
        if ((len(instruction) > 0)
                and (len(examples) > 0 or len(prompt) > 0)):
            instruction += self.instruction_suffix
        if ((len(examples) > 0 or len(histories) > 0)
                and (len(instruction.strip()) > 0 or len(self.example_list_prefix.strip()) > 0)):
            instruction += self.example_list_prefix
        if (len(examples) > 0 or len(histories) > 0) and len(prompt) > 0:
            prompt = self.example_input_prefix + prompt + self.example_input_suffix \
                + self.example_output_prefix
        return instruction + "".join(examples) + histories + prompt

    def _format_example(self, example: Example) -> str:
        """
        Formats the input/output of an example.
        """
        return self.example_input_prefix + example.input + self.example_input_suffix \
            + self.example_output_prefix + example.output + self.example_output_suffix

    def _format_histories(self, histories: List[Message]) -> str:
        """
        Formats the conversation histories as a list of input/output pairs.
        """
        if len(histories) % 2 != 0:
            raise ValueError("The number of messages must be even.")
        result = ""
        for i in range(0, len(histories), 2):
            result += self.example_input_prefix \
                + histories[i].content \
                + self.example_input_suffix \
                + self.example_output_prefix \
                + histories[i + 1].content \
                + self.example_output_suffix
        return result

    def load(self, config: Dict[str, str]) -> None:
        super().load(config)
        self.instruction_suffix = config.get("instruction_suffix",
                                             DEFAULT_INSTRUCTION_SUFFIX)
        self.example_list_prefix = config.get("example_list_prefix",
                                              DEFAULT_EXAMPLE_LIST_PREFIX)
        self.example_input_prefix = config.get("example_input_prefix",
                                               DEFAULT_EXAMPLE_INPUT_PREFIX)
        self.example_input_suffix = config.get("example_input_suffix",
                                               DEFAULT_EXAMPLE_INPUT_SUFFIX)
        self.example_output_prefix = config.get("example_output_prefix",
                                                DEFAULT_EXAMPLE_OUTPUT_PREFIX)
        self.example_output_suffix = config.get("example_output_suffix",
                                                DEFAULT_EXAMPLE_OUTPUT_SUFFIX)
