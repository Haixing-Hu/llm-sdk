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


class LatexCodeTextSplitter(RecursiveCharacterTextSplitter):
    """
    A text splitter which attempts to split the text along Latex-formatted
    layout elements.
    """

    def __init__(self, **kwargs: Any) -> None:
        separators = [
            # First, try to split along Latex sections
            "\n\\chapter{",
            "\n\\section{",
            "\n\\subsection{",
            "\n\\subsubsection{",
            # Now split by environments
            "\n\\begin{enumerate}",
            "\n\\begin{itemize}",
            "\n\\begin{description}",
            "\n\\begin{list}",
            "\n\\begin{quote}",
            "\n\\begin{quotation}",
            "\n\\begin{verse}",
            "\n\\begin{verbatim}",
            # Now split by math environments
            "\n\\begin{align}",
            "$$",
            "$",
            # Now split by the normal type of lines
            " ",
            "",
        ]
        super().__init__(separators=separators, **kwargs)
