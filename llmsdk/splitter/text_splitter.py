# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from abc import ABC, abstractmethod
from typing import Any, List, Callable
from logging import Logger, getLogger
from tqdm import tqdm

from ..common.document import Document
# from .text_splitter_utils import (
#     sort_splitted_documents,
#     check_original_document_id,
#     group_splitted_documents,
# )


class TextSplitter(ABC):

    def __init__(self, *,
                 chunk_size: int = 4000,
                 chunk_overlap: int = 200,
                 length_function: Callable[[str], int] = len,
                 show_progress: bool = False,
                 min_size_to_show_progress: int = 10) -> None:
        """
        Creates a new text splitter.

        :param chunk_size: the maximum number of basic unit in each splitted chunk.
        :param chunk_overlap: the number of basic units should overlap in each chunk.
        :param length_function: the function to calculate the length of the text
            in the basic unit.
        :param show_progress: indicates whether to show the progress of adding
            records.
        :param min_size_to_show_progress: the minimum number of records to show
            the progress.
        """
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._show_progress = show_progress
        self._min_size_to_show_progress = min_size_to_show_progress
        self._logger = getLogger(self.__class__.__name__)

    @property
    def show_progress(self) -> bool:
        return self._show_progress

    @show_progress.setter
    def show_progress(self, value: bool) -> None:
        self._show_progress = value

    @property
    def min_size_to_show_progress(self) -> int:
        return self._min_size_to_show_progress

    @min_size_to_show_progress.setter
    def min_size_to_show_progress(self, value: int) -> None:
        self._min_size_to_show_progress = value

    @property
    def logger(self) -> Logger:
        return self._logger

    def set_logging_level(self, level: int | str) -> None:
        """
        Sets the logging level of this object.

        :param level: the logging level to be set.
        """
        self._logger.setLevel(level)

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, value: int) -> None:
        self._chunk_size = value

    @property
    def chunk_overlap(self) -> int:
        return self._chunk_overlap

    @chunk_overlap.setter
    def chunk_overlap(self, value: int) -> None:
        self._chunk_overlap = value

    @property
    def length_function(self) -> Callable[[str], int]:
        return self._length_function

    @length_function.setter
    def length_function(self, value: Callable[[str], int]) -> None:
        self._length_function = value

    def _get_iterable(self, iterable: Any) -> Any:
        """
        Get an iterable or a tqdm progress bar.

        :param iterable: the iterable to be processed.
        :return: the iterable or the tqdm progress bar.
        """
        if self._show_progress and len(iterable) >= self._min_size_to_show_progress:
            return tqdm(iterable)
        else:
            return iterable

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """
        Splits a piece of text into a list of texts.

        :param text: the text to be split.
        :return: the list of texts splitted from the specified text.
        """

    def split_document(self, document: Document) -> List[Document]:
        """
        Splits a document into a list of documents.

        :param document: the document to be split.
        :return: the list of documents splitted from the specified document.
        """
        texts = self.split_text(document.content)
        if len(texts) == 1:
            return [document]
        else:
            result = []
            for index, text in enumerate(texts):
                doc = document.create_splitted_document(text, index)
                result.append(doc)
            return result

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of documents into a list of documents.

        :param documents: the list of documents to be split.
        :return: the list of documents splitted from the specified documents.
        """
        self._logger.info("Splitting %d documents...", len(documents))
        result = []
        for doc in self._get_iterable(documents):
            result.extend(self.split_document(doc))
        return result

    # @abstractmethod
    # def join_texts(self, texts: List[str]) -> str:
    #     """
    #     Joins the list of texts split from the same original text.
    #
    #     :param texts: the list of texts splitted from the same original text.
    #     :return: the original text.
    #     """
    #
    # def join_document(self, documents: List[Document]) -> Document:
    #     """
    #     Joins the split list of documents.
    #
    #     :param documents: the list of documents splitted from the same original
    #         document.
    #     :return: the original document.
    #     """
    #     if len(documents) == 0:
    #         raise ValueError("Empty document list.")
    #     if len(documents) == 1:
    #         return Document(id=documents[0].id,
    #                         content=documents[0].content,
    #                         metadata=documents[0].get_original_document_metadata())
    #     else:
    #         sorted_docs = sort_splitted_documents(documents)
    #         original_id = check_original_document_id(sorted_docs)
    #         texts = [doc.content for doc in sorted_docs]
    #         original_content = self.join_texts(texts)
    #         original_metadata = sorted_docs[0].get_original_document_metadata()
    #         return Document(id=original_id,
    #                         content=original_content,
    #                         metadata=original_metadata)
    #
    # def join_documents(self, documents: List[Document]) -> List[Document]:
    #     """
    #     Joins the split list of documents.
    #
    #     :param documents: the list of documents splitted from some original
    #         documents.
    #     :return: the original documents.
    #     """
    #     grouped_docs = group_splitted_documents(documents)
    #     result = []
    #     for key, value in grouped_docs.items():
    #         doc = self.join_document(value)
    #         result.append(doc)
    #     return result
