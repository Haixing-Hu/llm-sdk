# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from dataclasses import dataclass, field
from typing import Any, List

from .prompt_template import PromptTemplate
from ..common import Example, Role, Message


@dataclass
class ChatPromptTemplate(PromptTemplate):
    """
    The prompt template used to format the few-shot prompts.

    The formatted text prompt has the following form:

    ```
    [
        Message(Role.SYSTEM, {instruction}),
        Message(Role.HUMAN, {example1.input}),
        Message(Role.AI, {example1.output}),
        Message(Role.HUMAN, {example2.input}),
        Message(Role.AI, {example2.output}),
        Message(Role.HUMAN, {example3.input}),
        Message(Role.AI, {example3.output}),
        ...
        Message(Role.HUMAN, prompt),
    ]
    ```

    Examples:

    ```
        i1 = "Translate the following text into {language}."
        p1 = ChatPromptTemplate(instruction_template=i1, prompt_template="{prompt}")
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

        p2 = ChatPromptTemplate(instruction_template="{instruction}"，
                                prompt_template="{prompt}")
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

    examples: List[Example] = field(default_factory=list)
    """
    The list of examples.
    
    Note that the examples should not contain formatting placeholders.
    """

    def format(self, **kwargs: Any) -> List[Message]:
        result = []
        instruction = Message(Role.SYSTEM, self.instruction_template.format(**kwargs))
        if len(instruction.content) > 0:
            result.append(instruction)
        for e in self.examples:
            result.append(Message(Role.HUMAN, e.input))
            result.append(Message(Role.AI, e.output))
        prompt = Message(Role.HUMAN, self.prompt_template.format(**kwargs))
        result.append(prompt)
        return result
