# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import copy

from .metadata import Metadata


ORIGINAL_DOCUMENT_ID_ATTRIBUTE: str = "__original_document_id__"

ORIGINAL_DOCUMENT_INDEX_ATTRIBUTE: str = "__original_document_index__"


@dataclass
class Document:
    """
    The model represents documents.

    A document has a title, a content, and optional metadata.
    """

    content: str
    """The content of the document."""

    id: str = None
    """The ID of the document."""

    metadata: Optional[Metadata] = field(default_factory=dict)
    """The metadata of the document, or {} if no metadata."""

    def is_splitted(self) -> bool:
        """
        Tests whether this document is a splitted document.
        :return: True if this document is a splitted document; False otherwise.
        """
        return (self.metadata is not None) \
            and (ORIGINAL_DOCUMENT_ID_ATTRIBUTE in self.metadata) \
            and (ORIGINAL_DOCUMENT_INDEX_ATTRIBUTE in self.metadata)

    def get_original_document_id(self) -> str:
        """
        Gets the ID of the original document from the metadata of this splitted
        document.

        :return: the ID of the original document where this specified document
            is splitted from.
        :raise ValueError: if there is any error while getting the ID of the
            original document from the metadata of this splitted document.
        """
        if not self.is_splitted():
            raise ValueError("This document is not a splitted document.")
        result = self.metadata[ORIGINAL_DOCUMENT_ID_ATTRIBUTE]
        if result is None:
            raise ValueError(
                f"The document has an empty {ORIGINAL_DOCUMENT_ID_ATTRIBUTE} "
                f"attribute in its metadata: {self}"
            )
        if not isinstance(result, str):
            raise ValueError(
                f"The document has an invalid {ORIGINAL_DOCUMENT_ID_ATTRIBUTE} "
                f"attribute in its metadata: {self}"
            )
        return result

    def get_original_document_index(self) -> int:
        """
        Gets the index of this splitted document in its original document.

        :return: the index of this splitted document in its original document.
        :raise ValueError: if there is any error while getting the index of this
            splitted document in its original document.
        """
        if not self.is_splitted():
            raise ValueError("This document is not a splitted document.")
        result = self.metadata[ORIGINAL_DOCUMENT_INDEX_ATTRIBUTE]
        if result is None:
            raise ValueError(
                f"The document has an empty {ORIGINAL_DOCUMENT_INDEX_ATTRIBUTE} "
                f"attribute in its metadata: {self}"
            )
        if not isinstance(result, int):
            if isinstance(result, str):
                result = int(result)
            else:
                raise ValueError(
                    f"The document has an invalid {ORIGINAL_DOCUMENT_INDEX_ATTRIBUTE} "
                    f"attribute in its metadata: {self}"
                )
        return result

    def get_original_document_metadata(self) -> Metadata:
        """
        Gets the metadata of the original document from the metadata of this
        splitted document.

        :return: the metadata of the original document.
        """
        if not self.is_splitted():
            raise ValueError("This document is not a splitted document.")
        result = copy.deepcopy(self.metadata)
        result.pop(ORIGINAL_DOCUMENT_ID_ATTRIBUTE)
        result.pop(ORIGINAL_DOCUMENT_INDEX_ATTRIBUTE)
        return result

    @classmethod
    def create_splitted_document(cls,
                                 text: str,
                                 index: int,
                                 original_document: Document) -> Document:
        """
        Creates a splitted document.

        :param text: the splitted text of the content of the original document.
        :param index: the index of the splitted document.
        :param original_document: the original document.
        :return: the specified splitted document of the original document.
        """
        if (original_document.id is None) or (len(original_document.id) == 0):
            raise ValueError("The ID of the original document must be set.")
        id = original_document.id + "-" + str(index)
        metadata = copy.deepcopy(original_document.metadata)
        metadata[ORIGINAL_DOCUMENT_ID_ATTRIBUTE] = original_document.id
        metadata[ORIGINAL_DOCUMENT_INDEX_ATTRIBUTE] = index
        return Document(id=id, content=text, metadata=metadata)
