# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from dataclasses import dataclass
from typing import Any, List

from ..common import Role, Message
from .structured_prompt_template import StructuredPromptTemplate


@dataclass
class ChatPromptTemplate(StructuredPromptTemplate):
    """
    The prompt template used to format the few-shot prompts.

    The formatted prompt has the following form:

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
        p1 = ChatPromptTemplate(instruction_template=i1)
        p1.add_example(input="Hello, world!", output="你好，世界！")
        p1.add_example(input="What's your name?", output="你叫什么名字？")
        v1 = p1.format(language="Chinese", prompt="Today is Sunday.")
        print(f"v1={v1}")
        self.assertEqual([
            Message(Role.SYSTEM, "Translate the following text into Chinese."),
            Message(Role.HUMAN, "Hello, world!"),
            Message(Role.AI, "你好，世界！"),
            Message(Role.HUMAN, "What's your name?"),
            Message(Role.AI, "你叫什么名字？"),
            Message(Role.HUMAN, "Today is Sunday.")
        ], v1)

        p2 = ChatPromptTemplate("You are a helpful assistant.")
        p3.add_history(human_message="Who won the world series in 2020?",
                       ai_message="The Los Angeles Dodgers won the World Series in 2020.") in 2020.")
        v2 = p2.format(prompt="Where was it played?")
        print(f"v2={v2}")
        self.assertEqual([
            Message(Role.SYSTEM, "You are a helpful assistant."),
            Message(Role.HUMAN, "Who won the world series in 2020?"),
            Message(Role.AI, "The Los Angeles Dodgers won the World Series in 2020."),
            Message(Role.HUMAN, "Where was it played?")
        ], v2)
    ```

    """

    def format(self, **kwargs: Any) -> List[Message]:
        result = []
        instruction = Message(role=Role.SYSTEM,
                              content=self._format_instruction(**kwargs))
        if len(instruction.content) > 0:
            result.append(instruction)
        for e in self.examples:
            result.append(Message(Role.HUMAN, e.input))
            result.append(Message(Role.AI, e.output))
        result.extend(self.histories)
        prompt = Message(role=Role.HUMAN,
                         content=self._format_prompt(**kwargs))
        if len(prompt.content) > 0:
            result.append(prompt)
        return result
