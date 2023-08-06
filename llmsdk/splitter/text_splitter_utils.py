# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from typing import Iterable, List, Dict, Callable, Optional
import logging

from ..common.document import Document


def sort_splitted_documents(splitted_documents: List[Document]) -> List[Document]:
    """
    Sorts a list of document splitted from the same original document by their
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
        index = doc.get_splitted_document_index()
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
    original_id = splitted_documents[0].get_original_document_id()
    for i in range(1, n):
        doc_id = splitted_documents[i].get_original_document_id()
        if doc_id != original_id:
            raise ValueError(f"The splitted documents have different original IDs:\n"
                             f"{splitted_documents[0]}\n{splitted_documents[i]}")
    return original_id


def group_splitted_documents(documents: List[Document]) -> Dict[str, List[Document]]:
    """
    Groups the list of splitted documents by the IDs of their original documents.

    :param documents: a list of splitted documents.
    :return: A dictionary mapping the ID of original documents to the list of
        splitted documents from the same original document.
    """
    result: Dict[str, List[Document]] = {}
    for doc in documents:
        original_id = doc.get_original_document_id()
        if original_id in result:
            result[original_id].append(doc)
        else:
            result[original_id] = [doc]
    return result


def combine_splits(splits: Iterable[str],
                   separator: str,
                   chunk_size: int,
                   chunk_overlap: int,
                   length_function: Callable[[str], int]) -> List[str]:
    """
    Combines the smaller pieces of text into medium size chunks to send to
    the LLMs.

    :param splits: the list of splitted pieces of texts.
    :param separator: the separator used to combine texts.
    :param chunk_size: the maximum number of basic unit in each splitted chunk.
    :param chunk_overlap: the number of basic units should overlap in each chunk.
    :param length_function: the function to calculate the length of the text
        in the basic unit.
    :return: the list of medium size chunks of texts.
    """
    if chunk_overlap > chunk_size:
        raise ValueError(
            f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
            f"({chunk_size}), should be smaller."
        )
    seperator_len = length_function(separator)
    result: List[str] = []
    current: List[str] = []
    current_size = 0

    def next_size(new_size: int) -> int:
        """
        Calculates the total size if a new piece of text is added to the current
        list of texts.
        :param new_size: the size of the new piece of text.
        :return: the total size if a new piece of text is added to the current
            list of texts.
        """
        return current_size + new_size + (seperator_len if len(current) > 0 else 0)

    for split in splits:
        split_len = length_function(split)
        if next_size(split_len) > chunk_size:
            if current_size > chunk_size:
                logging.warning(f"Created a chunk of size {current_size}, "
                                f"which is longer than the specified {chunk_size}")
            if len(current) > 0:
                text = separator.join(current).strip()
                if len(text) > 0:
                    result.append(text)
                # Keep on popping if:
                # - we have a larger chunk than in the chunk overlap
                # - or if we still have any chunks and the length is long
                while (current_size > chunk_overlap) \
                        or ((current_size > 0) and (next_size(split_len) > chunk_size)):
                    current_size -= length_function(current[0]) \
                                    + (seperator_len if len(current) > 1 else 0)
                    current.pop(0)
        current_size = next_size(split_len)
        current.append(split)
    text = separator.join(current).strip()
    if len(text) > 0:
        result.append(text)
    return result
