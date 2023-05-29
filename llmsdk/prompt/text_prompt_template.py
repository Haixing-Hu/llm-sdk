# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from dataclasses import dataclass, field
from typing import Any, List

from .prompt_template import PromptTemplate
from ..common import Example


@dataclass
class TextPromptTemplate(PromptTemplate):
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

    prompt_template: str = ""
    """
    The template of the prompt of the final input.
    
    The template of prompt may contain formatting placeholders.
    """

    instruction_template: str = ""
    """
    The template of the instruction of the prompt.
    
    The template of instruction may contain formatting placeholders.
    """

    instruction_suffix: str = "\n\n"
    """
    The suffix for the instruction.
    """

    examples: List[Example] = field(default_factory=list)
    """
    The list of examples.
    
    Note that the examples should not contain formatting placeholders.
    """

    example_input_prefix: str = "input: "
    """
    The prefix for the input of an example.
    """

    example_input_suffix: str = "\n"
    """
    The suffix for the input of an example.
    """

    example_output_prefix: str = "output: "
    """
    The prefix for the output of an example.
    """

    example_output_suffix: str = "\n\n"
    """
    The suffix for the input of an example.
    """

    def format(self, **kwargs: Any) -> str:
        instruction = self.instruction_template.format(**kwargs)
        examples = [self._format_example(e) for e in self.examples]
        prompt = self.prompt_template.format(**kwargs)
        if (len(instruction) > 0) and (len(examples) > 0 or len(prompt) > 0):
            instruction += self.instruction_suffix
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
