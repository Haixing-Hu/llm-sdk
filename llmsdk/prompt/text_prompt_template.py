# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from dataclasses import dataclass
from typing import Any, Dict, ClassVar

from ..common import Example
from .structured_prompt_template import StructuredPromptTemplate


@dataclass
class TextPromptTemplate(StructuredPromptTemplate):
    """
    The prompt template used to format the few-shot prompts.

    The formatted text prompt has the following form:

    ```
    {instruction}{instruction_suffix}
    {example_input_prefix}{example1.input}{example_input_suffix}
    {example_output_prefix}{example1.output}{example_output_suffix}
    {example_input_prefix}{example2.input}{example_input_suffix}
    {example_output_prefix}{example2.output}{example_output_suffix}
    {example_input_prefix}{example3.input}{example_input_suffix}
    {example_output_prefix}{example3.output}{example_output_suffix}
    ...
    {example_input_prefix}{prompt}{example_input_suffix}
    {example_output_prefix}
    ```

    Examples:

    ```
        i1 = "Translate the following text into {language}."
        p1 = TextPromptTemplate(instruction_template=i1， prompt_template="{prompt}")
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

        p2 = TextPromptTemplate(instruction_template="{instruction}"，
                                prompt_template="{prompt}")
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
    ```

    """

    DEFAULT_INSTRUCTION_SUFFIX: ClassVar[str] = "\n"
    """
    The default suffix for the instruction.
    """

    DEFAULT_EXAMPLE_LIST_PREFIX = "\n"
    """
    The default prefix for the list of examples.
    """

    DEFAULT_EXAMPLE_INPUT_PREFIX: ClassVar[str] = "input: "
    """
    The default prefix for the input of an example.
    """

    DEFAULT_EXAMPLE_INPUT_SUFFIX: ClassVar[str] = "\n"
    """
    The default suffix for the input of an example.
    """

    DEFAULT_EXAMPLE_OUTPUT_PREFIX: ClassVar[str] = "output: "
    """
    The default prefix for the output of an example.
    """

    DEFAULT_EXAMPLE_OUTPUT_SUFFIX: ClassVar[str] = "\n\n"
    """
    The default suffix for the input of an example.
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
        instruction = self.instruction_template.format(**kwargs)
        examples = [self._format_example(e) for e in self.examples]
        prompt = self.prompt_template.format(**kwargs)
        if ((len(instruction) > 0)
                and (len(examples) > 0 or len(prompt) > 0)):
            instruction += self.instruction_suffix
        if ((len(examples) > 0)
                and (len(instruction.strip()) > 0
                     or len(self.example_list_prefix.strip()) > 0)):
            instruction += self.example_list_prefix
        if len(examples) > 0 and len(prompt) > 0:
            prompt = self.example_input_prefix + prompt + self.example_input_suffix \
                + self.example_output_prefix
        return instruction + "".join(examples) + prompt

    def _format_example(self, example: Example) -> str:
        """
        Formats the input/output of an example.
        """
        return self.example_input_prefix + example.input + self.example_input_suffix \
            + self.example_output_prefix + example.output + self.example_output_suffix

    def load(self, conf: Dict[str, str]) -> None:
        super().load(conf)
        self.instruction_suffix = conf.get(
            "instruction_suffix",
            TextPromptTemplate.DEFAULT_INSTRUCTION_SUFFIX
        )
        self.example_list_prefix = conf.get(
            "example_list_prefix",
            TextPromptTemplate.DEFAULT_EXAMPLE_LIST_PREFIX
        )
        self.example_input_prefix = conf.get(
            "example_input_prefix",
            TextPromptTemplate.DEFAULT_EXAMPLE_INPUT_PREFIX
        )
        self.example_input_suffix = conf.get(
            "example_input_suffix",
            TextPromptTemplate.DEFAULT_EXAMPLE_INPUT_SUFFIX
        )
        self.example_output_prefix = conf.get(
            "example_output_prefix",
            TextPromptTemplate.DEFAULT_EXAMPLE_OUTPUT_PREFIX
        )
        self.example_output_suffix = conf.get(
            "example_output_suffix",
            TextPromptTemplate.DEFAULT_EXAMPLE_OUTPUT_SUFFIX
        )
