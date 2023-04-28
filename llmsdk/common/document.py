# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import Any, Dict, Optional

from .metadata import Metadata


class Document:
    """
    The model represents documents.

    A document has a title, a content, and optional metadata.
    """

    def __init__(self,
                 content: str,
                 metadata: Optional[Metadata] = None) -> None:
        """
        Creates a Document object.

        :param content: the content of the document.
        :param metadata: the metadata of the document, or None if no metadata.
        """
        self._content = content
        self._metadata = {} if metadata is None else metadata

    @property
    def content(self) -> str:
        return self._content

    @content.setter
    def content(self, value: str) -> None:
        self._content = value

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    @metadata.setter
    def metadata(self, value: Optional[Metadata]) -> None:
        self._metadata = {} if value is None else value

    def dict(self) -> Dict[str, Any]:
        return {
            "content": self._content,
            "metadata": self._metadata,
        }

    def __str__(self) -> str:
        return str(self.dict())

    def __repr__(self) -> str:
        return str(self)
