# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from typing import Any

from .recursive_character_text_splitter import RecursiveCharacterTextSplitter


class MarkdownCodeTextSplitter(RecursiveCharacterTextSplitter):
    """
    A text splitter which attempts to split the text along Markdown-formatted
    headings.
    """

    def __init__(self, **kwargs: Any) -> None:
        separators = [
            # First, try to split along Markdown headings (starting with level 2)
            "\n## ",
            "\n### ",
            "\n#### ",
            "\n##### ",
            "\n###### ",
            # Note the alternative syntax for headings (below) is not handled here
            # Heading level 2
            # ---------------
            # End of code block
            "```\n\n",
            # Horizontal lines
            "\n\n***\n\n",
            "\n\n---\n\n",
            "\n\n___\n\n",
            # Note that this splitter doesn't handle horizontal lines defined
            # by *three or more* of ***, ---, or ___, but this is not handled
            "\n\n",
            "\n",
            " ",
            "",
        ]
        super().__init__(separators=separators, **kwargs)
