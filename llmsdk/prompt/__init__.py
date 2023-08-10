# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from .reply_parser import ReplyParser
from .dummy_reply_parser import DummyReplyParser
from .prompt_template import PromptTemplate
from .structured_prompt_template import (
    StructuredPromptTemplate,
    DEFAULT_INSTRUCTION_TEMPLATE,
    DEFAULT_CONTEXT_TEMPLATE,
    DEFAULT_OUTPUT_REQUIREMENT_TEMPLATE,
    DEFAULT_INPUT_TEMPLATE,
    DEFAULT_EXPLANATION_INSTRUCTION,
    DEFAULT_INSTRUCTION_PREFIX,
    DEFAULT_INSTRUCTION_SUFFIX,
    DEFAULT_CONTEXT_PREFIX,
    DEFAULT_CONTEXT_SUFFIX,
    DEFAULT_OUTPUT_REQUIREMENT_PREFIX,
    DEFAULT_OUTPUT_REQUIREMENT_SUFFIX,
    DEFAULT_EXPLANATION_INSTRUCTION_PREFIX,
    DEFAULT_EXPLANATION_INSTRUCTION_SUFFIX,
)
from .text_prompt_template import (
    TextPromptTemplate,
    DEFAULT_EXAMPLE_LIST_PREFIX,
    DEFAULT_EXAMPLE_LIST_SUFFIX,
    DEFAULT_EXAMPLE_INPUT_PREFIX,
    DEFAULT_EXAMPLE_INPUT_SUFFIX,
    DEFAULT_EXAMPLE_OUTPUT_PREFIX,
    DEFAULT_EXAMPLE_OUTPUT_SUFFIX,
)
from .chat_prompt_template import ChatPromptTemplate
