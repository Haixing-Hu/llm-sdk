# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import copy

from .metadata import Metadata


DOCUMENT_TYPE_ATTRIBUTE: str = "__type__"
"""
The name of the metadata attribute storing the type of a document. 
"""

ORIGINAL_DOCUMENT_ID_ATTRIBUTE: str = "__original_document_id__"
"""
The name of the metadata attribute storing the ID of the original document. 
"""

SPLITTED_DOCUMENT_INDEX_ATTRIBUTE: str = "__splitted_document_index__"
"""
The name of the metadata attribute storing the index of the splitted document. 
"""


@dataclass
class Document:
    """
    The model represents documents.

    A document has a title, a content, and optional metadata.
    """

    content: str
    """The content of the document."""

    metadata: Metadata = field(default_factory=Metadata)
    """The metadata of the document, or {} if no metadata."""

    id: str = None
    """The ID of the document."""

    score: Optional[float] = None
    """The score of this document relevant to the query."""

    def is_splitted(self) -> bool:
        """
        Tests whether this document is a splitted document.
        :return: True if this document is a splitted document; False otherwise.
        """
        return (self.metadata.has_value_of_type(ORIGINAL_DOCUMENT_ID_ATTRIBUTE, str)
                and self.metadata.has_value_of_type(SPLITTED_DOCUMENT_INDEX_ATTRIBUTE, int))

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
        return self.metadata[ORIGINAL_DOCUMENT_ID_ATTRIBUTE]

    def get_splitted_document_index(self) -> int:
        """
        Gets the index of this splitted document in its original document.

        :return: the index of this splitted document in its original document.
        :raise ValueError: if there is any error while getting the index of this
            splitted document in its original document.
        """
        if not self.is_splitted():
            raise ValueError("This document is not a splitted document.")
        return self.metadata[SPLITTED_DOCUMENT_INDEX_ATTRIBUTE]

    def get_original_document_metadata(self) -> Metadata:
        """
        Gets the metadata of the original document from the metadata of this
        splitted document.

        :return: the metadata of the original document.
        """
        if not self.is_splitted():
            raise ValueError("This document is not a splitted document.")
        metadata = copy.deepcopy(self.metadata)
        metadata.pop(ORIGINAL_DOCUMENT_ID_ATTRIBUTE)
        metadata.pop(SPLITTED_DOCUMENT_INDEX_ATTRIBUTE)
        return metadata

    def create_splitted_document(self,
                                 text: str,
                                 index: int) -> Document:
        """
        Creates a splitted document from this original document.

        :param text: the splitted text of the content of this original document.
        :param index: the index of the splitted document.
        :return: the specified splitted document of the original document.
        """
        if (self.id is None) or (len(self.id) == 0):
            raise ValueError(f"The ID of the original document must be set: {self}")
        id = self.id + "-" + str(index)
        metadata = copy.deepcopy(self.metadata)
        metadata[ORIGINAL_DOCUMENT_ID_ATTRIBUTE] = self.id
        metadata[SPLITTED_DOCUMENT_INDEX_ATTRIBUTE] = index
        return Document(id=id, content=text, metadata=metadata)

    @classmethod
    def from_record(cls,
                    id_field: str,
                    record: Dict[str, Any]) -> List[Document]:
        """
        Creates documents from a record.

        :param id_field: the name of the field storing the ID of the record.
        :param record: the record to be converted.
        :return: the list of documents converted from the specified record.
        """
        result = []
        metadata = Metadata(record)
        for key in record.keys():
            if key == id_field:
                continue
            content = str(record[key]).strip()
            if len(content) == 0:
                continue
            doc = Document(content=content, metadata=copy.deepcopy(metadata))
            result.append(doc)
        return result

    @classmethod
    def from_records(cls,
                     id_field: str,
                     records: List[Dict[str, Any]]) -> List[Document]:
        """
        Creates a list of documents from a list of records.

        :param id_field: the name of the field storing the ID of the record.
        :param records: the records to be converted.
        :return: the list of documents converted from the specified records.
        """
        result = []
        for record in records:
            result.extend(cls.from_record(id_field, record))
        return result
