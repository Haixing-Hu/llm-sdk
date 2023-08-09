# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from dataclasses import dataclass
from typing import Any, List

from ..common.message import Message
from ..common.role import Role
from .structured_prompt_template import StructuredPromptTemplate


@dataclass
class ChatPromptTemplate(StructuredPromptTemplate):
    """
    The prompt template used to format the few-shot prompts in the
    chat-completion models.

    The formatted prompt has the following form:

    ```
    [
        Message(Role.SYSTEM, formatted_instruction
                            + formatted_context
                            + formatted_output_indicator),
        Message(Role.HUMAN, examples[0].input),
        Message(Role.AI, examples[0].output),
        Message(Role.HUMAN, examples[1].input),
        Message(Role.AI, examples[1].output),
        Message(Role.HUMAN, examples[2].input),
        Message(Role.AI, examples[2].output),
        ...
        Message(Role.HUMAN, histories[0].content),
        Message(Role.AI, histories[1].content),
        Message(Role.HUMAN, histories[2].content),
        Message(Role.AI, histories[3].content),
        Message(Role.HUMAN, histories[4].content),
        Message(Role.AI, histories[5].content),
        ...
        Message(Role.HUMAN, formatted_input),
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
        instruction = self.format_instruction(**kwargs)
        context = self.format_context(**kwargs)
        output_requirement = self.format_output_requirement(**kwargs)
        if len(instruction) > 0 or len(context) > 0 or len(output_requirement) > 0:
            content = instruction + context + output_requirement
            result.append(Message(role=Role.SYSTEM, content=content.strip()))
        for e in self.examples:
            result.append(Message(role=Role.HUMAN, content=e.input.strip()))
            result.append(Message(role=Role.AI, content=e.output.strip()))
        result.extend(self.histories)
        input = self.format_input(**kwargs)
        if len(input) > 0:
            result.append(Message(role=Role.HUMAN, content=input.strip()))
        return result
