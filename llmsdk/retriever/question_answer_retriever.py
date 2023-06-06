# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import Any, List

from ..common import Document, Faq, SearchType
from ..vectorstore import VectorStore
from ..embedding import Embedding
from ..llm import LargeLanguageModel
from ..splitter import TextSplitter
from ..criterion import equal
from ..prompt import StructuredPromptTemplate
from .retriever import Retriever
from .vector_store_retriever import VectorStoreRetriever


class QuestionAnswerRetriever(Retriever):
    """
    A Question/Answer retriever based on a vector store and a LLM.
    """
    def __init__(self,
                 vector_store: VectorStore,
                 collection_name: str,
                 embedding: Embedding,
                 splitter: TextSplitter,
                 llm: LargeLanguageModel,
                 unknown_question_answer: str,
                 prompt_template: StructuredPromptTemplate,
                 direct_answer_score_threshold: float = 0.95,
                 question_score_threshold: float = 0.85,
                 answer_score_threshold: float = 0.85,
                 question_limit: int = 5,
                 answer_limit: int = 5) -> None:
        """
        Constructs a LlmQuestionAnswerRetriever.

        :param vector_store: the underlying vector store.
        :param collection_name: the name of the vector collection to use. The
            collection must store the vectors of the FAQs to be retrieved.
        :param embedding: the underlying embedding model.
        :param splitter: the text splitter used to split documents.
        :param llm: the underlying large language model.
        :param unknown_question_answer: the answer to be replied when the
            question of the user is unknown.
        :param prompt_template: the prompt template used to generate the prompt
            send to the LLM.
        :param direct_answer_score_threshold: the threshold of the scores of the
            direct answers. When user asks a question, the program will embed
            the question to a vector and search for the most similar question in
            the  predefined FAQs stored in the vector store. If the score of the
            question of a FAQ is greater than or equal to this threshold, the
            answer of that FAQ will be replied to the user directly.
        :param question_score_threshold: the threshold of the scores of the
            questions. When user asks a question, the program will embed the
            question to a vector and search for the most similar question in
            the predefined FAQs stored in the vector store. If the score of the
            question of a FAQ is greater than or equal to this threshold, the
            question and the answer of that FAQ will be selected as the context
            of the large language model.
        :param answer_score_threshold: the threshold of the scores of the answers.
            When user asks a question, the program will embed the question to a
            vector and search for the most similar question in the predefined
            FAQs stored in the vector store. If the score of the answer of a FAQ
            is greater than or equal to this threshold, the question and the
            answer of that FAQ will be selected as the context of the large
            language model.
        :param question_limit: the maximum number of the related questions to be
            selected.
        :param answer_limit: the maximum number of the related answers to be
            selected.
        """
        super().__init__()
        self._retriever = VectorStoreRetriever(vector_store=vector_store,
                                               collection_name=collection_name,
                                               embedding=embedding,
                                               splitter=splitter,
                                               search_type=SearchType.SIMILARITY)
        self._llm = llm
        self._unknown_question_answer = unknown_question_answer
        self._prompt_template = prompt_template
        self._direct_answer_score_threshold = direct_answer_score_threshold
        self._question_score_threshold = question_score_threshold
        self._answer_score_threshold = answer_score_threshold
        self._question_limit = question_limit
        self._answer_limit = answer_limit

    def open(self) -> None:
        self._retriever.open()

    def close(self) -> None:
        self._retriever.close()

    def ask(self, query: str) -> str:
        """
        Asks a question and gets the answer.

        :param query: the question to ask.
        :return: the answer of the question.
        """
        self._ensure_opened()
        # criterion to filter the questions of FAQs
        question_filter = equal(Document.FAQ_PROPERTY_ATTRIBUTE, "question")
        # criterion to filter the answers of FAQs
        answer_filter = equal(Document.FAQ_PROPERTY_ATTRIBUTE, "answer")
        self._logger.debug("Ask a question: %s", query)
        # search for the most similar questions in the FAQs
        question_docs = self._retriever.retrieve(
            query=query,
            limit=self._question_limit,
            score_threshold=self._question_score_threshold,
            criterion=question_filter
        )
        questions = Document.to_faqs(question_docs)
        if (len(questions) > 0
                and questions[0].score > self._direct_answer_score_threshold):
            # the score of the most similar question is greater than the
            # direct answer score threshold, so we can reply the answer
            # directly
            self._logger.debug("Direct answer: %s", questions[0].answer)
            return questions[0].answer
        answers_docs = self._retriever.retrieve(
            query=query,
            limit=self._answer_limit,
            score_threshold=self._answer_score_threshold,
            criterion=answer_filter
        )
        answers = Document.to_faqs(answers_docs)
        faqs = questions + answers
        if len(faqs) == 0:
            return self._unknown_question_answer
        # remove the duplicated elements in the faqs list
        faqs = list(set(faqs))
        # sort the faqs by their scores
        faqs.sort(key=lambda x: x.score, reverse=True)
        self._prompt_template.examples.clear()
        self._prompt_template.examples.extend(Faq.to_examples(faqs))
        self._logger.debug("Prompt template: %s", self._prompt_template)
        # generate the prompt
        prompt = self._prompt_template.format(
            question=query,
            unknown_question_answer=self._unknown_question_answer
        )
        self._logger.debug("Prompt: %s", prompt)
        # generate the answer by the LLM
        answer = self._llm.generate(prompt)
        return answer

    def retrieve(self, query: str, **kwargs: Any) -> List[Document]:
        answer = self.ask(query)
        return [Document(content=answer)]
