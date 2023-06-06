# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, List, ClassVar
import json

from ..common import Example
from .prompt_template import PromptTemplate


@dataclass
class StructuredPromptTemplate(PromptTemplate, ABC):
    """
    The interface of a structured prompt templates.
    """

    DEFAULT_PROMPT_TEMPLATE: ClassVar[str] = ""
    """
    The default template of the prompt of the final input.
    """

    DEFAULT_INSTRUCTION_TEMPLATE: ClassVar[str] = ""
    """
    The default template of the instruction of the prompt.
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

    def load(self, conf: Dict[str, Any]) -> None:
        """
        Loads the configuration of this prompt template.

        :param conf: the dictionary of the configuration.
        """
        self.instruction_template = conf.get(
            "instruction_template",
            StructuredPromptTemplate.DEFAULT_INSTRUCTION_TEMPLATE
        )
        self.prompt_template = conf.get(
            "prompt_template",
            StructuredPromptTemplate.DEFAULT_PROMPT_TEMPLATE
        )
        self.examples.clear()
        if "examples" in conf:
            for e in conf["examples"]:
                id = e.get("id", None)
                input = e.get("input")
                output = e.get("output")
                example = Example(id=id, input=input, output=output)
                self.examples.append(example)
