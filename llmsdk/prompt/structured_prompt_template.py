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

DEFAULT_INSTRUCTION_TEMPLATE: str = "{instruction}"
"""
The default template of the specific task or instruction you want the model to 
perform.
"""

DEFAULT_CONTEXT_TEMPLATE: str = "{context}"
"""
The default template of the external information or additional context that can 
steer the model to better responses.
"""

DEFAULT_OUTPUT_REQUIREMENT_TEMPLATE: str = "{output_requirement}"
"""
The default template of the requirement of the type or format of the output.
"""

DEFAULT_INPUT_TEMPLATE: str = "{input}"
"""
The default template of the input or question that we are interested to find a 
response for.
"""

DEFAULT_INSTRUCTION_PREFIX: str = ""
"""
The default prefix for the instruction.
"""

DEFAULT_INSTRUCTION_SUFFIX: str = "\n\n"
"""
The default suffix for the instruction.
"""

DEFAULT_CONTEXT_PREFIX: str = "The following are known context:\n"
"""
The default prefix for the context.
"""

DEFAULT_CONTEXT_SUFFIX: str = "\n\n"
"""
The default suffix for the context.
"""

DEFAULT_OUTPUT_REQUIREMENT_PREFIX: str = \
    "The output must satisfy the following requirements:\n"
"""
The default prefix for the output requirement.
"""

DEFAULT_OUTPUT_REQUIREMENT_SUFFIX: str = "\n\n"
"""
The default suffix for the output requirement.
"""


@dataclass
class StructuredPromptTemplate(PromptTemplate, ABC):
    """
    The base class of structured prompt templates.

    A structured prompt template is a template that can be used to format the
      few-shot prompts. It has the following elements:

    - Instruction: a specific task or instruction you want the model to perform.
    - Context: external information or additional context that can steer the
        model to better responses.
    - Output Requirement: the requirement of the type or format of the output.
    - Examples: the list of examples that the model can learn from.
    - Input: the input or question that we are interested to find a response for.

    You do not need all the elements for a prompt and the format depends on the
    task at hand.
    """

    instruction_template: str = DEFAULT_INSTRUCTION_TEMPLATE
    """
    The template of the specific task or instruction you want the model to 
    perform.

    This template may contain formatting placeholders.
    """

    context_template: str = DEFAULT_CONTEXT_TEMPLATE
    """
    The template of the external information or additional context that can
    steer the model to better responses.
    
    This template may contain formatting placeholders.
    """

    output_requirement_template: str = DEFAULT_OUTPUT_REQUIREMENT_TEMPLATE
    """
    The template of the requirement of the type or format of the output.
    
    This template may contain formatting placeholders.
    """

    input_template: str = DEFAULT_INPUT_TEMPLATE
    """
    The template of the input or question that we are interested to find a 
    response for.

    This template may contain formatting placeholders.
    """

    instruction_prefix: str = DEFAULT_INSTRUCTION_PREFIX
    """
    The prefix for the instruction.
    """

    instruction_suffix: str = DEFAULT_INSTRUCTION_SUFFIX
    """
    The suffix for the instruction.
    """

    context_prefix: str = DEFAULT_CONTEXT_PREFIX
    """
    The prefix for the context.
    """

    context_suffix: str = DEFAULT_CONTEXT_SUFFIX
    """
    The suffix for the context.
    """

    output_requirement_prefix: str = DEFAULT_OUTPUT_REQUIREMENT_PREFIX
    """
    The prefix for the output requirement.
    """

    output_requirement_suffix: str = DEFAULT_OUTPUT_REQUIREMENT_SUFFIX
    """
    The suffix for the output requirement.
    """

    examples: List[Example] = field(default_factory=list)
    """
    The list of examples that the model can learn from.
    
    Note that the examples should not contain formatting placeholders.
    """

    histories: List[Message] = field(default_factory=list)
    """
    The list of conversation histories.
    
    Note that the histories should not contain formatting placeholders.
    """

    formatted_instruction: str = ""
    """
    The formatted instruction.
    """

    formatted_context: str = ""
    """
    The formatted context.
    """

    formatted_output_requirement: str = ""
    """
    The formatted output requirement.
    """

    formatted_input: str = ""
    """
    The formatted input.
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
        context_template = config.get("context_template",
                                      DEFAULT_CONTEXT_TEMPLATE)
        output_requirement_template = config.get("output_requirement_template",
                                                 DEFAULT_OUTPUT_REQUIREMENT_TEMPLATE)
        input_template = config.get("input_template",
                                    DEFAULT_INPUT_TEMPLATE)
        instruction_prefix = config.get("instruction_prefix",
                                        DEFAULT_INSTRUCTION_PREFIX)
        instruction_suffix = config.get("instruction_suffix",
                                        DEFAULT_INSTRUCTION_SUFFIX)
        context_prefix = config.get("context_prefix",
                                    DEFAULT_CONTEXT_PREFIX)
        context_suffix = config.get("context_suffix",
                                    DEFAULT_CONTEXT_SUFFIX)
        output_requirement_prefix = config.get("output_requirement_prefix",
                                               DEFAULT_OUTPUT_REQUIREMENT_PREFIX)
        output_requirement_suffix = config.get("output_requirement_suffix",
                                               DEFAULT_OUTPUT_REQUIREMENT_SUFFIX)
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
        self.instruction_prefix = instruction_prefix
        self.instruction_suffix = instruction_suffix
        self.context_template = context_template
        self.context_prefix = context_prefix
        self.context_suffix = context_suffix
        self.output_requirement_template = output_requirement_template
        self.output_requirement_prefix = output_requirement_prefix
        self.output_requirement_suffix = output_requirement_suffix
        self.input_template = input_template
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
        """
        Checks whether the specified list of histories is valid.

        :param histories: the specified list of histories.
        :raises ValueError: if the specified list of histories is invalid.
        """
        if len(histories) % 2 != 0:
            raise ValueError("The number of conversation histories must be even.")
        for i in range(0, len(histories), 2):
            if histories[i].role != Role.HUMAN:
                raise ValueError("The message at index {} is not a human "
                                 "message.".format(i))
            if histories[i + 1].role != Role.AI:
                raise ValueError("The message at index {} is not an AI "
                                 "message.".format(i + 1))

    def format_instruction(self, **kwargs: Any) -> str:
        """
        Formats the instruction of this template.

        This function will set the attribute `formatted_instruction` of this
        template to the formatted instruction.

        :param kwargs: the keyword arguments to be used to format the
            instruction.
        :return: the formatted instruction.
        """
        if (self.instruction_template == DEFAULT_INSTRUCTION_TEMPLATE
                and ("instruction" not in kwargs)):
            result = ""
        else:
            result = self.instruction_template.format(**kwargs)
        if len(result) > 0:
            result = self.instruction_prefix + result + self.instruction_suffix
        self.formatted_instruction = result
        return result

    def format_context(self, **kwargs: Any) -> str:
        """
        Formats the context of this template.

        This function will set the attribute `formatted_context` of this
        template to the formatted context.

        :param kwargs: the keyword arguments to be used to format the context.
        :return: the formatted context.
        """
        if (self.context_template == DEFAULT_CONTEXT_TEMPLATE
                and ("context" not in kwargs)):
            result = ""
        else:
            result = self.context_template.format(**kwargs)
        if len(result) > 0:
            result = self.context_prefix + result + self.context_suffix
        self.formatted_context = result
        return result

    def format_output_requirement(self, **kwargs: Any) -> str:
        """
        Formats the output requirement of this template.

        This function will set the attribute `formatted_output_requirement` of
        this template to the formatted output requirement.

        :param kwargs: the keyword arguments to be used to format the output
            requirement.
        :return: the formatted output requirement.
        """
        if (self.output_requirement_template == DEFAULT_OUTPUT_REQUIREMENT_TEMPLATE
                and ("output_requirement" not in kwargs)):
            result = ""
        else:
            result = self.output_requirement_template.format(**kwargs)
        if len(result) > 0:
            result = self.output_requirement_prefix + result + self.output_requirement_suffix
        self.formatted_output_requirement = result
        return result

    def format_input(self, **kwargs: Any) -> str:
        """
        Formats the input of this template.

        This function will set the attribute `formatted_input` of this template
        to the formatted input.

        :param kwargs: the keyword arguments to be used to format the input.
        :return: the formatted input.
        """
        if (self.input_template == DEFAULT_INPUT_TEMPLATE
                and ("input" not in kwargs)):
            result = ""
        else:
            result = self.input_template.format(**kwargs)
        self.formatted_input = result
        return result
