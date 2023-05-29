# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import copy
from typing import List, Dict, Optional

from ..common import Metadata, Document

ORIGINAL_DOCUMENT_ID_ATTRIBUTE: str = "__original_document_id__"

ORIGINAL_DOCUMENT_INDEX_ATTRIBUTE: str = "__original_document_index__"


def create_splitted_document(text: str,
                             index: int,
                             original_document: Document) -> Document:
    """
    Creates a splitted document.

    :param text: the splitted text of the content of the original document.
    :param index: the index of the splitted document.
    :param original_document: the original document.
    :return: the specified splitted document of the original document.
    """
    id = original_document.id + "-" + str(index)
    metadata = copy.deepcopy(original_document.metadata)
    metadata[ORIGINAL_DOCUMENT_ID_ATTRIBUTE] = original_document.id
    metadata[ORIGINAL_DOCUMENT_INDEX_ATTRIBUTE] = index
    return Document(id=id, content=text, metadata=metadata)


def get_original_document_id(splitted_document: Document) -> str:
    """
    Gets the ID of the original document from the metadata of a splitted document.

    :param splitted_document: a splitted document.
    :return: the ID of the original document where the specified document is
        splitted from.
    :raise ValueError: if there is any error while getting the ID of the
        original document from the metadata of a splitted document.
    """
    metadata = splitted_document.metadata
    if metadata is None:
        raise ValueError(f"No metadata for the document: {splitted_document}")
    if ORIGINAL_DOCUMENT_ID_ATTRIBUTE not in metadata:
        raise ValueError(f"The document has no {ORIGINAL_DOCUMENT_ID_ATTRIBUTE} "
                         f"attribute in its metadata: {splitted_document}")
    original_doc_id = metadata[ORIGINAL_DOCUMENT_ID_ATTRIBUTE]
    if original_doc_id is None:
        raise ValueError(f"The document has an empty {ORIGINAL_DOCUMENT_ID_ATTRIBUTE} "
                         f"attribute in its metadata: {splitted_document}")
    if not isinstance(original_doc_id, str):
        raise ValueError(f"The document has an invalid {ORIGINAL_DOCUMENT_ID_ATTRIBUTE} "
                         f"attribute in its metadata: {splitted_document}")
    return original_doc_id


def get_original_document_index(splitted_document: Document) -> int:
    """
    Gets the index of a splitted document in its original document.

    :param splitted_document: a splitted document.
    :return: the index of a splitted document in its original document.
    :raise ValueError: if there is any error while getting the index of the
        splitted document in its original document.
    """
    metadata = splitted_document.metadata
    if metadata is None:
        raise ValueError(f"No metadata for the document: {splitted_document}")
    if ORIGINAL_DOCUMENT_INDEX_ATTRIBUTE not in metadata:
        raise ValueError(f"The document has no {ORIGINAL_DOCUMENT_INDEX_ATTRIBUTE} "
                         f"attribute in its metadata: {splitted_document}")
    original_doc_index = metadata[ORIGINAL_DOCUMENT_INDEX_ATTRIBUTE]
    if original_doc_index is None:
        raise ValueError(f"The document has an empty {ORIGINAL_DOCUMENT_INDEX_ATTRIBUTE} "
                         f"attribute in its metadata: {splitted_document}")
    if not isinstance(original_doc_index, int):
        if isinstance(original_doc_index, str):
            original_doc_index = int(original_doc_index)
        else:
            raise ValueError(f"The document has an invalid {ORIGINAL_DOCUMENT_INDEX_ATTRIBUTE} "
                             f"attribute in its metadata: {splitted_document}")
    return original_doc_index


def get_ordered_splitted_documents(splitted_documents: List[Document]) -> List[Document]:
    """
    Order a list of document splitted from the same original document by their
    index in the original document.

    :param splitted_documents: a list of document splitted from the same
        original document
    :return: the list of splitted document ordered by their index in their
        original document.
    :raise ValueError: if there is any error while sorting the splitted documents.
    """
    n = len(splitted_documents)
    if n <= 0:
        raise ValueError("Empty list of splitted documents.")
    result: List[Optional[Document]] = [None] * n
    for doc in splitted_documents:
        index = get_original_document_index(doc)
        if index < 0 or index >= n:
            raise ValueError(f"Invalid splitted index of the document: {doc}")
        if result[index] is not None:
            raise ValueError(f"Duplicated splitted document index:\n"
                             f"{result[index]}\n{doc}")
        result[index] = doc
    return result


def check_original_document_id(splitted_documents: List[Document]) -> str:
    """
    Checks the ID of the original document of a list of splitted documents.

    :param splitted_documents: a list of splitted document which should be
        splitted from the same original document.
    :return: the ID of the same original document of the list of splitted
        documents.
    :raise ValueError: if the list of splitted document is not splitted from the
        same original document.
    """
    n = len(splitted_documents)
    if n <= 0:
        raise ValueError("Empty list of splitted documents.")
    original_id = get_original_document_id(splitted_documents[0])
    for i in range(1, n):
        doc_id = get_original_document_id(splitted_documents[i])
        if doc_id != original_id:
            raise ValueError(f"The splitted documents have different original IDs:\n"
                             f"{splitted_documents[0]}\n{splitted_documents[i]}")
    return original_id


def get_original_metadata(splitted_document: Document) -> Metadata:
    """
    Gets the metadata of the original document from the metadata of the splitted
    document.

    :param splitted_document: a splitted document.
    :return: the metadata of the original document.
    """
    metadata = splitted_document.metadata
    if metadata is None:
        raise ValueError(f"No metadata for the document: {splitted_document}")
    result = copy.deepcopy(metadata)
    result.pop(ORIGINAL_DOCUMENT_ID_ATTRIBUTE)
    result.pop(ORIGINAL_DOCUMENT_INDEX_ATTRIBUTE)
    return result


def group_splitted_documents(documents: List[Document]) -> Dict[str, List[Document]]:
    """
    Groups the list of splitted documents by the IDs of their original documents.

    :param documents: a list of splitted documents.
    :return: A dictionary mapping the ID of original documents to the list of
        splitted documents from the same original document.
    """
    result: Dict[str, List[Document]] = {}
    for doc in documents:
        original_id = get_original_document_id(doc)
        if original_id in result:
            result[original_id].append(doc)
        else:
            result[original_id] = [doc]
    return result
