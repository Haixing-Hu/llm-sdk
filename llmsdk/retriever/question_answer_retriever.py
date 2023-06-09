# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
from typing import Any, List, Dict, Optional

from ..common import Document, Faq, SearchType
from ..vectorstore import VectorStore, CollectionInfo
from ..embedding import Embedding
from ..llm import LargeLanguageModel, ModelType
from ..splitter import TextSplitter
from ..criterion import equal
from ..prompt import StructuredPromptTemplate, TextPromptTemplate, ChatPromptTemplate
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
                 answer_limit: Optional[int] = None) -> None:
        """
        Constructs a LlmQuestionAnswerRetriever.

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
        self.__init_parameters()

    def __load_config(self) -> Dict[str, Any]:
        """
        Loads the default configuration with respect to the specified langauge.

        :return: the default configuration with respect to the specified langauge.
        """
        match self._language:
            case "en_US":
                from .conf.question_answer_retriever__en_US import CONFIG
                return CONFIG
            case "zh_CN":
                from .conf.question_answer_retriever__zh_CN import CONFIG
                return CONFIG
            case _:
                raise ValueError(f"Unsupported language: {self._language}")

    def __init_parameters(self) -> None:
        if self._default_config is None:
            config = self.__load_config()
            self._default_config = config
        else:
            config = self._default_config
        if not self._unknown_question_answer:
            self._unknown_question_answer = config["unknown_question_answer"]
        if not self._prompt_template:
            match self._llm.model_type:
                case ModelType.TEXT_COMPLETION:
                    self._prompt_template = TextPromptTemplate()
                case ModelType.CHAT_COMPLETION:
                    self._prompt_template = ChatPromptTemplate()
                case _:
                    raise ValueError(f"Unsupported LLM model type: {self._llm.model_type}")
            self._prompt_template.load(config["prompt_template"])
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

    def _open(self, **kwargs: Any) -> None:
        self._retriever.open(**kwargs)
        self._is_opened = True

    def _close(self) -> None:
        self._retriever.close()
        self._is_opened = False

    def ask(self, query: str) -> str:
        """
        Asks a question and gets the answer.

        :param query: the question to ask.
        :return: the answer of the question.
        """
        self._logger.info("The user asks a question: '%s'", query)
        self._ensure_opened()
        answer = self._ask(query)
        self._logger.info("Get the following answer: '%s'", answer)
        return answer

    def _ask(self, query: str) -> str:
        """
        Asks a question and gets the answer.

        :param query: the question to ask.
        :return: the answer of the question.
        """
        # criterion to filter the questions of FAQs
        question_filter = equal(Document.FAQ_PROPERTY_ATTRIBUTE, "question")
        # criterion to filter the answers of FAQs
        answer_filter = equal(Document.FAQ_PROPERTY_ATTRIBUTE, "answer")
        # search for the most similar questions in the FAQs
        question_docs = self._retriever.retrieve(
            query=query,
            limit=self._question_limit,
            score_threshold=self._question_score_threshold,
            criterion=question_filter
        )
        questions = Document.to_faqs(question_docs)
        self._logger.info("Found %d similar questions: %s", len(questions), questions)
        if (len(questions) > 0
                and questions[0].score > self._direct_answer_score_threshold):
            # the score of the most similar question is greater than the
            # direct answer score threshold, so we can reply the answer
            # directly
            self._logger.info("The score of the most similar question is %f,"
                              "which is greater than the direct answer score "
                              "threshold %f: %s",
                              questions[0].score,
                              self._direct_answer_score_threshold,
                              questions[0].answer)
            return questions[0].answer
        answers_docs = self._retriever.retrieve(
            query=query,
            limit=self._answer_limit,
            score_threshold=self._answer_score_threshold,
            criterion=answer_filter
        )
        answers = Document.to_faqs(answers_docs)
        self._logger.info("Found %d related answers: %s", len(answers), answers)
        faqs = questions + answers
        if len(faqs) == 0:
            return self._unknown_question_answer
        # remove the duplicated elements in the faqs list
        faqs = list(set(faqs))
        # sort the faqs by their scores
        faqs.sort(key=lambda x: x.score, reverse=True)
        self._logger.info("Get %d different related FAQs: %s", len(faqs), faqs)
        self._prompt_template.examples.clear()
        self._prompt_template.examples.extend(Faq.to_examples(faqs))
        self._logger.debug("The prompt template is: %s", self._prompt_template)
        # generate the prompt
        prompt = self._prompt_template.format(
            question=query,
            unknown_question_answer=self._unknown_question_answer
        )
        self._logger.info("The prompt is: %s", prompt)
        # generate the answer by the LLM
        answer = self._llm.generate(prompt)
        return answer

    def _retrieve(self, query: str, **kwargs: Any) -> List[Document]:
        answer = self._ask(query)
        return [Document(content=answer)]

    def add(self, faq: Faq) -> List[Document]:
        """
        Adds a FAQ to this retriever.

        :param faq: the FAQ to add.
        :return: the list of actual documents added to this retriever, which may
            be the sub-documents splitted from the original document.
        """
        self._logger.info("Adding a FAQ to the retriever %s ...",
                          self._retriever_name)
        self._logger.debug("The FAQ to add is: %s", faq)
        self._ensure_opened()
        docs = Document.from_faq(faq)
        self._logger.debug("The FAQ is converted into %d documents: %s",
                           len(docs), docs)
        return self._retriever.add_all(docs)

    def add_all(self, faqs: List[Faq]) -> List[Document]:
        """
        Adds a list of FAQs to this retriever.

        :param faqs: the list of FAQs to add.3
        :return: the list of actual documents added to this retriever, which may
            be the sub-documents splitted from the original document.
        """
        self._logger.info("Adding a list of FAQs to the retriever %s ...",
                          self._retriever_name)
        self._logger.debug("The FAQs to add are: %s", faqs)
        self._ensure_opened()
        docs = Document.from_faqs(faqs)
        self._logger.debug("The FAQs are converted into %d documents: %s",
                           len(docs), docs)
        return self._retriever.add_all(docs)

    def get_store_info(self) -> CollectionInfo:
        """
        Gets the information of the collection of underlying vector store.

        :return: the information of the collection of underlying vector store.
        """
        return self._retriever.get_store_info()
