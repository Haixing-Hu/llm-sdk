# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, List
import json

from ..common.example import Example
from ..common.message import Message
from ..common.role import Role
from .prompt_template import PromptTemplate


DEFAULT_PROMPT_TEMPLATE: str = "{prompt}"
"""
The default template of the prompt of the final input.
"""

DEFAULT_INSTRUCTION_TEMPLATE: str = "{instruction}"
"""
The default template of the instruction of the prompt.
"""


@dataclass
class StructuredPromptTemplate(PromptTemplate, ABC):
    """
    The interface of a structured prompt templates.
    """

    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    """
    The template of the prompt of the final input.

    The template of prompt may contain formatting placeholders.
    """

    instruction_template: str = DEFAULT_INSTRUCTION_TEMPLATE
    """
    The template of the instruction of the prompt.

    The template of instruction may contain formatting placeholders.
    """

    examples: List[Example] = field(default_factory=list)
    """
    The list of examples.
    
    Note that the examples should not contain formatting placeholders.
    """

    histories: List[Message] = field(default_factory=list)
    """
    The list of conversation histories.
    
    Note that the histories should not contain formatting placeholders.
    """

    def load_from_file(self, file_path: str) -> None:
        """
        Loads the configuration of this prompt template from a file in the JSON
        format.

        :param file_path: the path of the configuration file in the JSON format.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            conf = json.loads(text)
            self.load(conf)

    def load(self, config: Dict[str, Any]) -> None:
        """
        Loads the configuration of this prompt template.

        :param config: the dictionary of the configuration.
        """
        instruction_template = config.get("instruction_template",
                                          DEFAULT_INSTRUCTION_TEMPLATE)
        prompt_template = config.get("prompt_template",
                                     DEFAULT_PROMPT_TEMPLATE)
        examples = []
        if "examples" in config:
            for e in config["examples"]:
                id = e.get("id", None)
                input = e.get("input")
                output = e.get("output")
                example = Example(id=id, input=input, output=output)
                examples.append(example)
        histories = []
        if "histories" in config:
            data = config["histories"]
            if len(data) % 2 != 0:
                raise ValueError("The number of conversation histories must be even.")
            for i in range(0, len(data), 2):
                if data[i]["role"] != Role.HUMAN.value:
                    raise ValueError("The message at index {} is not a human "
                                     "message.".format(i))
                if data[i + 1]["role"] != Role.AI.value:
                    raise ValueError("The message at index {} is not an AI "
                                     "message.".format(i + 1))
                histories.append(Message(Role.HUMAN, data[i]["content"]))
                histories.append(Message(Role.AI, data[i + 1]["content"]))
        # Avoid destroy the content of this object if the above statements raise
        #   any exception.
        self.instruction_template = instruction_template
        self.prompt_template = prompt_template
        self.examples = examples
        self.histories = histories

    def clear_examples(self) -> None:
        """
        Clears all examples.
        """
        self.examples.clear()

    def add_example(self, input: str, output: str) -> None:
        """
        Adds an example.

        :param input: the input of the added example.
        :param output: the output of the added example.
        """
        self.examples.append(Example(input=input, output=output))

    def add_examples(self, examples: List[Example]) -> None:
        """
        Adds a list of examples.

        :param examples: the list of examples to be added.
        """
        self.examples.extend(examples)

    def set_examples(self, examples: List[Example]) -> None:
        """
        Sets the examples of this template to the specified example list.

        :param examples: the specified example list.
        """
        self.examples.clear()
        self.examples.extend(examples)

    def clear_histories(self) -> None:
        """
        Clears all histories.
        """
        self.histories.clear()

    def add_history(self, human_message: str, ai_message: str) -> None:
        """
        Adds a piece of conversation history.

        :param human_message: the message said by the human.
        :param ai_message: the message replied by the AI.
        """
        self.histories.append(Message(role=Role.HUMAN, content=human_message))
        self.histories.append(Message(role=Role.AI, content=ai_message))

    def add_histories(self, histories: List[Message]) -> None:
        """
        Adds a list of histories.

        :param histories: the list of conversation histories to be added, which
        must alternat human and AI messages.
        """
        self._check_histories(histories)
        for i in range(0, len(histories), 2):
            self.histories.append(histories[i])
            self.histories.append(histories[i + 1])

    def set_histories(self, histories: List[Message]) -> None:
        """
        Sets the histories of this template to the specified history list.

        :param histories: the specified history list.
        """
        self._check_histories(histories)
        self.histories.clear()
        self.histories.extend(histories)

    def _check_histories(self, histories: List[Message]) -> None:
        if len(histories) % 2 != 0:
            raise ValueError("The number of conversation histories must be even.")
        for i in range(0, len(histories), 2):
            if histories[i].role != Role.HUMAN:
                raise ValueError("The message at index {} is not a human "
                                 "message.".format(i))
            if histories[i + 1].role != Role.AI:
                raise ValueError("The message at index {} is not an AI "
                                 "message.".format(i + 1))

    def _format_instruction(self, **kwargs: Any) -> str:
        if (self.instruction_template == DEFAULT_INSTRUCTION_TEMPLATE
                and ("instruction" not in kwargs)):
            return ""
        else:
            return self.instruction_template.format(**kwargs)

    def _format_prompt(self, **kwargs: Any) -> str:
        if (self.prompt_template == DEFAULT_PROMPT_TEMPLATE
                and ("prompt" not in kwargs)):
            return ""
        else:
            return self.prompt_template.format(**kwargs)
