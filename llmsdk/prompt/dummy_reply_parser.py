# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from typing import Optional, Any

from ..common.prompt import Prompt
from .reply_parser import ReplyParser


class DummyReplyParser(ReplyParser):
    """
    The class of dummy reply parsers which do not parse the reply.
    """

    def parse(self, reply: str, prompt: Optional[Prompt] = None) -> Any:
        return reply
