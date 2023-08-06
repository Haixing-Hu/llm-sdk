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


class PythonCodeTextSplitter(RecursiveCharacterTextSplitter):
    """
    A text splitter which attempts to split the text along Python syntax.
    """

    def __init__(self, **kwargs: Any) -> None:
        separators = [
            # First, try to split along class definitions
            "\nclass ",
            "\ndef ",
            "\n\tdef ",
            # Now split by the normal type of lines
            "\n\n",
            "\n",
            " ",
            "",
        ]
        super().__init__(separators=separators, **kwargs)
