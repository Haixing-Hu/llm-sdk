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
    DEFAULT_PROMPT_TEMPLATE,
)
from .text_prompt_template import (
    TextPromptTemplate,
    DEFAULT_EXAMPLE_INPUT_PREFIX,
    DEFAULT_EXAMPLE_INPUT_SUFFIX,
    DEFAULT_EXAMPLE_LIST_PREFIX,
    DEFAULT_EXAMPLE_OUTPUT_PREFIX,
    DEFAULT_EXAMPLE_OUTPUT_SUFFIX,
    DEFAULT_INSTRUCTION_SUFFIX,
)
from .chat_prompt_template import ChatPromptTemplate
