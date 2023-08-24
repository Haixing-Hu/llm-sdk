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
import json

from ..common.search_type import SearchType
from ..common.document import Document, RECORD_FIELD_ATTRIBUTE
from ..vectorstore.vector_store import VectorStore
from ..embedding.embedding import Embedding
from ..llm.llm import LargeLanguageModel
from ..splitter.text_splitter import TextSplitter
from ..criterion.criterion_builder import equal
from ..prompt.structured_prompt_template import StructuredPromptTemplate
from ..util.common_utils import record_to_csv, records_to_csv
from .vector_store_based_retriever import VectorStoreBasedRetriever


class SimilarRecordRetriever(VectorStoreBasedRetriever):
    """
    A retriever that retrieves semantically similar records from a list of
    known records.
    """
    def __init__(self, *,
                 record_id_field: str,
                 vector_store: VectorStore,
                 collection_name: str,
                 embedding: Embedding,
                 splitter: TextSplitter,
                 llm: LargeLanguageModel,
                 default_config: Optional[Dict[str, Any]] = None,
                 language: Optional[str] = "en_US",
                 prompt_template: Optional[StructuredPromptTemplate] = None,
                 record_limit: Optional[int] = None,
                 record_score_threshold: Optional[float] = None,
                 use_cache: bool = True,
                 cache_size: int = 10000,
                 show_progress: bool = False,
                 min_size_to_show_progress: int = 10) -> None:
        """
        Constructs a `SimilarRecordRetriever`.

        :param record_id_field: the name of the field storing the ID of records.
        :param vector_store: the underlying vector store.
        :param collection_name: the name of the vector collection to use. The
            collection must store the vectors of the list of known records.
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
        :param prompt_template: the prompt template used to generate the prompt
            send to the LLM. If this argument is set to `None`, the class will
            use the default prompt template from the default configuration.
        :param record_limit: the maximum number of the related records to be
            selected. If this argument is set to `None`, the class will use the
            default value from the default configuration.
        :param record_score_threshold: the threshold of the scores of the
            similar records. When user gives a query record, the program will
            embed the record to a vector and search for the most similar records
            in the known list of records stored in the vector store. If the
            score of a known record is greater than or equal to this threshold,
            the record will be selected as the context of the underlying large
            language model. If this argument is set to `None`, the class will
            use the default value from the default configuration.
        :param use_cache: indicates whether to use the cache to store the
            embedded vectors of texts. If this argument is True, the embedded
            vectors of texts will be cached in a LRU cache. Otherwise, the
            embedded vectors of texts will not be cached.
        :param cache_size: the number of text embeddings to be cached. This
            argument is ignored if the use_cache argument is False.
        :param show_progress: indicates whether to show the progress of adding
            records.
        :param min_size_to_show_progress: the minimum number of records to show
            the progress.
        """
        super().__init__(vector_store=vector_store,
                         collection_name=collection_name,
                         embedding=embedding,
                         splitter=splitter,
                         llm=llm,
                         search_type=SearchType.SIMILARITY,
                         use_cache=use_cache,
                         cache_size=cache_size,
                         show_progress=show_progress,
                         min_size_to_show_progress=min_size_to_show_progress)
        self._record_id_field = record_id_field
        self._default_config = default_config
        self._language = language
        self._prompt_template = prompt_template
        self._record_limit = record_limit
        self._record_score_threshold = record_score_threshold
        self._histories = {"explanation": "No query."}
        self.__init_parameters()

    def __init_parameters(self) -> None:
        if self._default_config is None:
            module = f".conf.similar_record_retriever__{self._language}"
            config = import_module(name=module, package=__package__).CONFIG
        else:
            config = self._default_config

        if not self._prompt_template:
            model_type = self._llm.model_type
            template_cfg = config["prompt_template"]
            self._prompt_template = model_type.load_prompt_template(template_cfg)

        if not self._record_limit:
            self._record_limit = config["record_limit"]

        if not self._record_score_threshold:
            self._record_score_threshold = config["record_score_threshold"]

    def add_record(self, record: Dict[str, Any]) -> List[Document]:
        """
        Adds a known record to this retriever.

        :param record: the known record to be added.
        :return: the list of actual documents added to this retriever, which may
            be the sub-documents split from the original document.
        """
        self._logger.info("Adding a record to the retriever %s ...",
                          self._retriever_name)
        self._logger.debug("The record to add is: %s", record)
        self._ensure_opened()
        docs = Document.from_record(self._record_id_field, record)
        self._logger.debug("The record is converted into %d documents: %s",
                           len(docs), docs)
        return self._retriever.add_all(docs)

    def add_records(self, records: List[Dict[str, Any]]) -> List[Document]:
        """
        Adds a list of known records to this retriever.

        :param records: the list of known records to add.
        :return: the list of actual documents added to this retriever, which may
            be the sub-documents split from the original document.
        """
        self._logger.info("Adding a list of records to the retriever %s ...",
                          self._retriever_name)
        self._logger.debug("The records to add are: %s", records)
        self._ensure_opened()
        self._logger.info("Constructing documents from records...")
        docs = []
        for record in self._get_iterable(records):
            docs.extend(Document.from_record(self._record_id_field, record))
        self._logger.debug("The records are converted into %d documents: %s",
                           len(docs), docs)
        return self._retriever.add_all(docs)

    def find(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Finds the most similar record in the known list of records to the given
        query record.

        :param record: the query record.
        :return: the most similar record in the known list of records to the
            given query record, or `None` if no similar record is found.
        """
        self._logger.info("Finding the most similar record to the query record "
                          "in the retriever %s ...", self._retriever_name)
        self._logger.debug("The query record is: %s", record)
        self._ensure_opened()
        result = self._find(record)
        self._logger.debug("The most similar documents are: %s", result)
        return result

    def _find(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Finds the most similar record in the known list of records to the given
        query record.

        :param record: the query record.
        :return: the most similar record in the known list of records to the
            given query record, or `None` if no similar record is found.
        """
        similar_records = self.__get_top_similar_records(record)
        if len(similar_records) == 0:
            self._histories = {
                "explanation": "No similar record found in the vector database.",
            }
            return None
        self._logger.debug("The top similar records are: %s", similar_records)
        id_field = self._record_id_field
        self._histories = {
            "known_records": records_to_csv(similar_records),
            "query_record": record_to_csv(record),
            "id_field": id_field,
            "explanation": "",
        }
        prompt = self._prompt_template.format_prompt(
            known_records=self._histories["known_records"],
            query_record=self._histories["query_record"],
            id_field=id_field
        )
        self._histories["prompt"] = prompt
        self._logger.info("The prompt to LLM is:\n%s", prompt)
        reply = self._llm.generate(prompt).strip()
        self._logger.info("The answer from LLM is: %s", reply)
        self._histories["reply"] = reply
        try:
            result = json.loads(reply)
        except json.JSONDecodeError:
            self._logger.error("Failed to parse the JSON reply from LLM: %s", reply)
            return None
        self._histories["answer"] = result["answer"]
        self._histories["explanation"] = result["explanation"]
        if result["answer"] == "NONE":
            self._logger.info("No similar record found by the LLM.")
            return None
        else:
            for r in similar_records:
                if (id_field in r) and (r[id_field] == result["answer"]):
                    return r
            self._logger.warn("The LLM returns a record that is not in the "
                              "top similar records: %s", reply)
            return None

    def __get_top_similar_records(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Gets the list of similar records to the given query record.

        :param record: the query record.
        :return: the list of similar records to the given query record.
        """
        result = []
        for key in record:
            docs = self._retriever.retrieve(
                query=str(record[key]),
                limit=self._record_limit,
                score_threshold=self._record_score_threshold,
                criterion=equal(RECORD_FIELD_ATTRIBUTE, key),
            )
            result.extend(Document.to_records(self._record_id_field, docs))
        if len(result) == 0:
            # try to find the similar records without the attribute constraint
            self._logger.info("No similar records are found with the attribute "
                              "constraint, trying to find similar records "
                              "without the attribute constraint ...")
            for key in record:
                docs = self._retriever.retrieve(
                    query=str(record[key]),
                    limit=self._record_limit,
                    score_threshold=self._record_score_threshold,
                )
                result.extend(Document.to_records(self._record_id_field, docs))
        self._logger.info("Found %d similar records: %s", len(result), result)
        return result

    def _retrieve(self, query: str, **kwargs: Any) -> List[Document]:
        record = self._find({"query": query})
        if record is None:
            return []
        else:
            return Document.from_record(self._record_id_field, record)

    def explain(self) -> str:
        """
        Gets the explanation of the last query.

        :return: the explanation of the last query.
        """
        if len(self._histories["explanation"]) > 0:
            return self._histories["explanation"]
        else:
            explanation_prompt = self._prompt_template.format_explanation_prompt(
                last_reply=self._histories["reply"],
                known_records=self._histories["known_records"],
                query_record=self._histories["query_record"],
                id_field=self._histories["id_field"],
            )
            self._logger.info("The explanation prompt to LLM is:\n%s", explanation_prompt)
            explanation = self._llm.generate(explanation_prompt).strip()
            self._logger.info("The explanation from LLM is: %s", explanation)
            self._histories["explanation"] = explanation
            return explanation
