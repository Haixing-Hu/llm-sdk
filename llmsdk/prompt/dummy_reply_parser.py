# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import Optional, Any

from .reply_parser import ReplyParser
from ..common import Prompt


class DummyReplyParser(ReplyParser):
    """
    The class of dummy reply parsers which do not parse the reply.
    """

    def parse(self, reply: str, prompt: Optional[Prompt] = None) -> Any:
        return reply
