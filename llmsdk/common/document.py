# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import Dict, Optional


class Document:
    """
    The model represents documents.

    A document has a title, a content, and optional metadata.
    """

    def __init__(self,
                 content: str,
                 title: Optional[str] = None,
                 metadata: Optional[Dict[str, str]] = None) -> None:
        """
        Creates a Document object.

        :param title: the title of the document, or None if no title.
        :param content: the content of the document.
        :param metadata: the metadata of the document, or None if no metadata.
        """
        self._content = content
        self._title = title
        self._metadata = metadata

    @property
    def title(self) -> Optional[str]:
        return self._title

    @title.setter
    def title(self, value: Optional[str]) -> None:
        self._title = value

    @property
    def content(self) -> str:
        return self._content

    @content.setter
    def content(self, value: str) -> None:
        self._content = value

    @property
    def metadata(self) -> Optional[Dict[str, str]]:
        return self._metadata

    @metadata.setter
    def metadata(self, value: Optional[Dict[str, str]]) -> None:
        self._metadata = value

    def dict(self) -> Dict[str, str | Dict[str, str]]:
        return {
            "title": self._title,
            "content": self._content,
            "metadata": self._metadata,
        }

    def __str__(self) -> str:
        return str(self.dict())

    def __repr__(self) -> str:
        return str(self.dict())
