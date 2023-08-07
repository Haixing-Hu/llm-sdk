# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
from typing import Any, List, Dict, Optional
from importlib import import_module

from ..common.role import Role
from ..common.message import Message
from ..common.search_type import SearchType
from ..common.faq import Faq, FAQ_PART_ATTRIBUTE
from ..common.document import Document
from ..vectorstore.collection_info import CollectionInfo
from ..vectorstore.vector_store import VectorStore
from ..embedding.embedding import Embedding
from ..llm.llm import LargeLanguageModel
from ..splitter.text_splitter import TextSplitter
from ..criterion.criterion_builder import equal
from ..prompt.structured_prompt_template import StructuredPromptTemplate
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
                 default_config: Optional[Dict[str, Any]] = None,
                 language: Optional[str] = "en_US",
                 unknown_question_answer: Optional[str] = None,
                 prompt_template: Optional[StructuredPromptTemplate] = None,
                 direct_answer_score_threshold: Optional[float] = None,
                 question_score_threshold: Optional[float] = None,
                 answer_score_threshold: Optional[float] = None,
                 question_limit: Optional[int] = None,
                 answer_limit: Optional[int] = None,
                 history_limit: Optional[int] = None) -> None:
        """
        Constructs a `QuestionAnswerRetriever`.

        :param vector_store: the underlying vector store.
        :param collection_name: the name of the vector collection to use. The
            collection must store the vectors of the FAQs to be retrieved.
        :param embedding: the underlying embedding model.
        :param splitter: the text splitter used to split documents.
        :param llm: the underlying large language model.
        :param default_config: the default configuration of this retriever. If
            this argument is not `None`, the class will use items in this
            configuration as default values to initialize the parameters of this
            class; Otherwise, the class will load the predefined default
            configuration with respect to the language specified by the
            `language` argument.
        :param language: the language of the predefined default configuration.
            Default value is "en_US".
        :param unknown_question_answer: the answer to be replied when the
            question of the user is unknown. If this argument is set to `None`,
            the class will use the default value from the default configuration.
        :param prompt_template: the prompt template used to generate the prompt
            send to the LLM. If this argument is set to `None`, the class will
            use the default prompt template from the default configuration.
        :param direct_answer_score_threshold: the threshold of the scores of the
            direct answers. When user asks a question, the program will embed
            the question to a vector and search for the most similar question in
            the  predefined FAQs stored in the vector store. If the score of the
            question of a FAQ is greater than or equal to this threshold, the
            answer of that FAQ will be replied to the user directly. If this
            argument is set to `None`, the class will use the default value from
            the default configuration.
        :param question_score_threshold: the threshold of the scores of the
            questions. When user asks a question, the program will embed the
            question to a vector and search for the most similar question in
            the predefined FAQs stored in the vector store. If the score of the
            question of a FAQ is greater than or equal to this threshold, the
            question and the answer of that FAQ will be selected as the context
            of the large language model. If this argument is set to `None`, the
            class will use the default value from the default configuration.
        :param answer_score_threshold: the threshold of the scores of the answers.
            When user asks a question, the program will embed the question to a
            vector and search for the most similar question in the predefined
            FAQs stored in the vector store. If the score of the answer of a FAQ
            is greater than or equal to this threshold, the question and the
            answer of that FAQ will be selected as the context of the large
            language model. If this argument is set to `None`, the class will
            use the default value from the default configuration.
        :param question_limit: the maximum number of the related questions to be
            selected. If this argument is set to `None`, the class will use the
            default value from the default configuration.
        :param answer_limit: the maximum number of the related answers to be
            selected. If this argument is set to `None`, the class will use the
            default value from the default configuration.
        :param history_limit: the maximum number of the remembered conversation
            histories. If this argument is set to `None`, the class will use the
            default value from the default configuration.
        """
        super().__init__()
        self._retriever = VectorStoreRetriever(vector_store=vector_store,
                                               collection_name=collection_name,
                                               embedding=embedding,
                                               splitter=splitter,
                                               search_type=SearchType.SIMILARITY)
        self._llm = llm
        self._default_config = default_config
        self._language = language
        self._unknown_question_answer = unknown_question_answer
        self._prompt_template = prompt_template
        self._direct_answer_score_threshold = direct_answer_score_threshold
        self._question_score_threshold = question_score_threshold
        self._answer_score_threshold = answer_score_threshold
        self._question_limit = question_limit
        self._answer_limit = answer_limit
        self._history_limit = history_limit
        self._histories: List[Message] = []
        self.__init_parameters()

    def __init_parameters(self) -> None:
        if self._default_config is None:
            module = f".conf.question_answer_retriever__{self._language}"
            config = import_module(name=module, package=__package__).CONFIG
        else:
            config = self._default_config
        if not self._unknown_question_answer:
            self._unknown_question_answer = config["unknown_question_answer"]
        if not self._prompt_template:
            self._prompt_template = self._llm.model_type.load_prompt_template(config["prompt_template"])
        if not self._direct_answer_score_threshold:
            self._direct_answer_score_threshold = config["direct_answer_score_threshold"]
        if not self._question_score_threshold:
            self._question_score_threshold = config["question_score_threshold"]
        if not self._answer_score_threshold:
            self._answer_score_threshold = config["answer_score_threshold"]
        if not self._question_limit:
            self._question_limit = config["question_limit"]
        if not self._answer_limit:
            self._answer_limit = config["answer_limit"]
        if not self._history_limit:
            self._history_limit = config["history_limit"]

    def get_store_info(self) -> CollectionInfo:
        """
        Gets the information of the collection of underlying vector store.

        :return: the information of the collection of underlying vector store.
        """
        return self._retriever.get_store_info()

    def _open(self, **kwargs: Any) -> None:
        self._retriever.open(**kwargs)
        self._is_opened = True

    def _close(self) -> None:
        self._retriever.close()
        self._is_opened = False

    def add_faq(self, faq: Faq) -> List[Document]:
        """
        Adds a FAQ to this retriever.

        :param faq: the FAQ to add.
        :return: the list of actual documents added to this retriever, which may
            be the sub-documents split from the original document.
        """
        self._logger.info("Adding a FAQ to the retriever %s ...",
                          self._retriever_name)
        self._logger.debug("The FAQ to add is: %s", faq)
        self._ensure_opened()
        docs = Faq.to_document(faq)
        self._logger.debug("The FAQ is converted into %d documents: %s",
                           len(docs), docs)
        return self._retriever.add_all(docs)

    def add_faqs(self, faqs: List[Faq]) -> List[Document]:
        """
        Adds a list of FAQs to this retriever.

        :param faqs: the list of FAQs to add.
        :return: the list of actual documents added to this retriever, which may
            be the sub-documents split from the original document.
        """
        self._logger.info("Adding a list of FAQs to the retriever %s ...",
                          self._retriever_name)
        self._logger.debug("The FAQs to add are: %s", faqs)
        self._ensure_opened()
        docs = Faq.to_documents(faqs)
        self._logger.debug("The FAQs are converted into %d documents: %s",
                           len(docs), docs)
        return self._retriever.add_all(docs)

    def add_document(self, doc: Document) -> List[Document]:
        """
        Adds a document to this retriever.

        :param doc: the document to add.
        :return: the list of actual documents added to this retriever, which may
            be the sub-documents split from the original document.
        """
        self._logger.info("Adding a document to the retriever %s ...",
                          self._retriever_name)
        self._logger.debug("The document to add is: %s", doc)
        self._ensure_opened()
        return self._retriever.add(doc)

    def add_documents(self, docs: List[Document]) -> List[Document]:
        """
        Adds a list of documents to this retriever.

        :param docs: the list of documents to add.
        :return: the list of actual documents added to this retriever, which may
            be the sub-documents split from the original document.
        """
        self._logger.info("Adding a list of documents to the retriever %s ...",
                          self._retriever_name)
        self._logger.debug("The documents to add are: %s", docs)
        self._ensure_opened()
        return self._retriever.add_all(docs)

    def ask(self, question: str) -> str:
        """
        Asks a question and gets the answer.

        :param question: the question to ask.
        :return: the answer of the question.
        """
        self._logger.info("The user asks a question: '%s'", question)
        self._ensure_opened()
        answer = self._ask(question)
        self._logger.info("Get the following answer: '%s'", answer)
        self.__append_history(question, answer)
        return answer

    def __append_history(self, question: str, answer: str) -> None:
        """
        Adds a history of a question and its answer to the remembered histories.

        If the length of remembered histories exceeds the limit, the oldest
        history will be forgotten.

        :param question: the question asked by the user.
        :param answer: the answer replied by the AI.
        """
        self._logger.debug("Adding a history of a question and its answer to "
                           "the remembered histories: '%s' -> '%s'",
                           question, answer)
        if len(self._histories) >= self._history_limit * 2:
            self._histories.pop(0)
            self._histories.pop(0)
        self._histories.append(Message(Role.HUMAN, question))
        self._histories.append(Message(Role.AI, answer))

    def _ask(self, question: str) -> str:
        """
        Asks a question and gets the answer.

        :param question: the question to ask.
        :return: the answer of the question.
        """
        question_faqs = self.__get_similar_questions(question)
        if (len(question_faqs) > 0
                and question_faqs[0].score > self._direct_answer_score_threshold):
            # the score of the most similar question is greater than the
            # direct answer score threshold, so we can reply the answer
            # directly
            self._logger.info("The score of the most similar question is %f,"
                              "which is greater than the direct answer score "
                              "threshold %f: %s",
                              question_faqs[0].score,
                              self._direct_answer_score_threshold,
                              question_faqs[0])
            return question_faqs[0].answer
        answer_faqs = self.__get_related_answers(question)
        faqs = question_faqs + answer_faqs
        if len(faqs) == 0:
            return self._unknown_question_answer
        # remove the duplicated elements in the faqs list
        faqs = list(set(faqs))
        # sort the faqs by their scores
        faqs.sort(key=lambda x: x.score, reverse=True)
        self._logger.info("Get %d different related FAQs: %s",
                          len(faqs),
                          [(q.question, q.score) for q in faqs])
        self._prompt_template.set_examples(Faq.to_examples(faqs))
        self._prompt_template.set_histories(self._histories)
        self._logger.debug("The prompt template is:\n%s", self._prompt_template)
        # generate the prompt
        prompt = self._prompt_template.format(
            question=question,
            unknown_question_answer=self._unknown_question_answer
        )
        self._logger.info("The prompt is:\n%s", prompt)
        # generate the answer by the LLM
        answer = self._llm.generate(prompt)
        return answer

    def __get_similar_questions(self, question: str) -> List[Faq]:
        self._logger.info("Searching the similar FAQ questions to: %s", question)
        # criterion to filter the questions of FAQs
        criterion = equal(FAQ_PART_ATTRIBUTE, "question")
        # search for the most similar questions in the FAQs
        docs = self._retriever.retrieve(
            query=question,
            limit=self._question_limit,
            score_threshold=self._question_score_threshold,
            criterion=criterion
        )
        result = Faq.from_documents(docs)
        self._logger.info("Found %d similar questions: %s",
                          len(result),
                          [(q.question, q.score) for q in result])
        return result

    def __get_related_answers(self, question: str) -> List[Faq]:
        self._logger.info("Searching the related FAQ answer to: %s", question)
        # criterion to filter the answers of FAQs
        criterion = equal(FAQ_PART_ATTRIBUTE, "answer")
        docs = self._retriever.retrieve(
            query=question,
            limit=self._answer_limit,
            score_threshold=self._answer_score_threshold,
            criterion=criterion
        )
        result = Faq.from_documents(docs)
        self._logger.info("Found %d related FAQ answers: %s",
                          len(result),
                          [(q.question, q.score) for q in result])
        return result

    def _retrieve(self, query: str, **kwargs: Any) -> List[Document]:
        answer = self._ask(query)
        return [Document(content=answer)]
