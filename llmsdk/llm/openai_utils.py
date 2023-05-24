# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import logging
import os
from typing import Any, Callable, Dict, List, Optional
from dataclasses import asdict

import openai
import tiktoken
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..common import ChatMessage
from ..util.common_utils import (
    singleton,
    read_config_file,
    is_website_accessible,
)

MODEL_TOKEN_MAPPING = {
    # GPT-4 models: https://platform.openai.com/docs/models/gpt-4
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0314": 32768,
    # GPT-3.5 models: https://platform.openai.com/docs/models/gpt-3-5
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0301": 4096,
    "text-davinci-003": 4097,
    "text-davinci-002": 4097,
    # GPT-3 models: https://platform.openai.com/docs/models/gpt-3
    "text-curie-001": 2049,
    "text-babbage-001": 2040,
    "text-ada-001": 2049,
    "davinci": 2049,
    "curie": 2049,
    "babbage": 2049,
    "ada": 2049,
    # codex models: https://platform.openai.com/docs/models/codex
    "code-davinci-002": 8001,
    "code-davinci-001": 8001,
    "code-cushman-002": 2048,
    "code-cushman-001": 2048,
    # embedding models: https://platform.openai.com/docs/guides/embeddings/second-generation-models
    "text-embedding-ada-002": 8191,
}
DEFAULT_MAX_RETRIES = 6
DEFAULT_WAIT_MIN_SECONDS = 4
DEFAULT_WAIT_MAX_SECONDS = 10

# the logger of the current module
logger = logging.getLogger(__name__)


def call_with_retries(openai_api: Callable[[Any], Any],
                      max_retries: int = DEFAULT_MAX_RETRIES,
                      wait_min_seconds: int = DEFAULT_WAIT_MIN_SECONDS,
                      wait_max_seconds: int = DEFAULT_WAIT_MAX_SECONDS,
                      **kwargs: Any) -> Any:
    """
    Call an OpenAI's API with retries.

    :param openai_api: the OpenAI's API to call.
    :param max_retries: the maximum number of retries.
    :param wait_min_seconds: the minimum seconds to wait before each retry.
    :param wait_max_seconds: the maximum seconds to wait before each retry.
    :param kwargs: the arguments passed to the OpenAI's API.
    :return: the result of calling the OpenAI's API.
    """

    @retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=wait_min_seconds, max=wait_max_seconds),
        retry=(
            retry_if_exception_type(openai.error.Timeout)
            | retry_if_exception_type(openai.error.APIError)
            | retry_if_exception_type(openai.error.APIConnectionError)
            | retry_if_exception_type(openai.error.TryAgain)
            | retry_if_exception_type(openai.error.ServiceUnavailableError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def __submit_openai_request() -> Any:
        return openai_api(**kwargs)

    return __submit_openai_request()


def get_model_tokens(model) -> int:
    """
    Gets the context length in tokens of the specified model.

    :param model: the name of the OpenAI's model.
    :return: the context length in tokens of the specified model.
    :raise ValueError: if the maximum number of tokens of the model is unknown.
    """
    result = MODEL_TOKEN_MAPPING[model]
    if result is None:
        raise ValueError(f"The maximum number of tokens of the model {model} "
                         f"is unknown.")
    return result


def count_tokens(model, text) -> int:
    """
    Counts the number of tokens of the specified text encoded by the specified
    model.

    :param model: the name of the OpenAI's model.
    :param text: the specified text.
    :return: the number of tokens of the specified text encoded by the model.
    """
    codec = tiktoken.encoding_for_model(model)
    tokenized_text = codec.encode(text)
    return len(tokenized_text)


def count_message_tokens(model: str,
                         messages: List[ChatMessage]):
    """
    Counts the number of tokens used by a list of chatting messages.

    Reference: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

    :param messages: list of chatting messages.
    :param model: the name of OpenAI model.
    """
    if model == "gpt-3.5-turbo":
        logger.warning("gpt-3.5-turbo may change over time. Returning num tokens "
                       "assuming gpt-3.5-turbo-0301.")
        return count_message_tokens(model="gpt-3.5-turbo-0301",
                                    messages=messages)
    elif model == "gpt-4":
        logger.warning("gpt-4 may change over time. Returning num tokens "
                       "assuming gpt-4-0314.")
        return count_message_tokens(model="gpt-4-0314",
                                    messages=messages)
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""
                ChatGpt._count_tokens_of_messages() is not implemented for 
                model {model}. 
                See https://github.com/openai/openai-python/blob/main/chatml.md 
                for information on how messages are converted to tokens.
            """)
    codec = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        msg_dict = asdict(message)
        for key, value in msg_dict.items():
            num_tokens += len(codec.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def check_model_compatibility(model: str, endpoint: str) -> None:
    """
    Checks the model compatibility with the specified API endpoint.

    see: https://platform.openai.com/docs/models/model-endpoint-compatibility

    :param model: the name of the model.
    :param endpoint: the name of the API endpoint.
    :raise ValueError: if the model is not compatible with the specified API endpoint.
    """
    match endpoint:
        case "chat-completions":
            compatible_models = [
                "gpt-4",
                "gpt-4-0314",
                "gpt-4-32k",
                "gpt-4-32k-0314",
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-0301",
            ]
        case "completions":
            compatible_models = [
                "text-davinci-003",
                "text-davinci-002",
                "text-curie-001",
                "text-babbage-001",
                "text-ada-001"
            ]
        case "edits":
            compatible_models = [
                "text-davinci-edit-001",
                "code-davinci-edit-001",
            ]
        case "audio-transcriptions":
            compatible_models = [
                "whisper-1",
            ]
        case "audio-translations":
            compatible_models = [
                "whisper-1",
            ]
        case "fine-tunes":
            compatible_models = [
                "davinci",
                "curie",
                "babbage",
                "ada",
            ]
        case "embeddings":
            compatible_models = [
                "text-embedding-ada-002",
                # "text-search-ada-doc-001", # deprecate the V1 API
            ]
        case "moderations":
            compatible_models = [
                "text-moderation-stable",
                "text-moderation-latest",
            ]
        case _:
            raise ValueError(f"Unsupported API endpoint: '{endpoint}'")
    if model not in compatible_models:
        raise ValueError(f"Model '{model}' is incompatible with the endpoint "
                         f"'{endpoint}'. "
                         f"The compatible models are: {compatible_models}")


def get_chunked_tokens(model: str,
                       text: str) -> List[List[int]]:
    """
    Split a long text into chunks according to the maximum number of tokens of
    the model, and returns the list of chunked token list.

    :param model: the name of the specified model.
    :param text: the long text to be split.
    :return: a list of token list, each token list has the length not exceed the
        maximum number of tokens of the model.
    """
    # encode the text to list of tokens
    codec = tiktoken.encoding_for_model(model)
    tokens = codec.encode(text)
    # Here we simply divide the input text into chunks by their maximum allowed
    # length.
    # FIXME: In some cases, it may make sense to split chunks on paragraph
    #  boundaries or sentence boundaries to help preserve the meaning of the
    #  text.
    chunk_size = get_model_tokens(model)
    return [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]


def set_api_key(api_key: Optional[str]) -> None:
    """
    Sets the API key of OpenAI.

    If the API key is not provided by the argument 'key', the program will try
    to get it from the environment variable 'OPENAI_API_KEY'. If it is also not set,
    the program will try to get it from the configuration file. The path of the
    configuration file should be set in the environment variable "OPENAI_CONFIG_PATH".
    If the environment variable "OPENAI_CONFIG_FILE" is not set, the program
    will use the default configuration file path "~/.openai/config", where "~"
    is the alias of the home directory of the current user, depending on the
    current operating system.

    :param api_key: the provided API key, or None if not provided.
    :return: the OpenAI's API key, or None if not set.
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        config_file = os.getenv("OPENAI_CONFIG_PATH")
        if config_file is None:
            config_file = os.path.join(os.path.expanduser("~"), ".openai", "config")
        config = read_config_file(config_file)
        api_key = config["OPENAI_API_KEY"]
    if api_key is None:
        raise ValueError("The OpenAI's API key is not set. "
                         "It should be set either by the argument 'api_key', "
                         "or by the environment variable 'OPENAI_API_KEY', "
                         "or in the configuration file '~/.openai/config'.")
    logger.info("Setting API key of OpenAI...")
    openai.api_key = api_key


def set_proxy(proxy: Optional[Dict] = None) -> dict:
    """
    Sets the proxy used by the OpenAI's API.

    :param proxy: the configuration of proxy to set, which should be a dict. For
    example: {"http": "<PROXY>", "https": "<PROXY>"}. If it is None, the
    program will use the proxy set in the environment variables "http_proxy"
    and "https_proxy".
    """
    if proxy is None:
        proxy = {}
    if ("http_proxy" not in proxy) or (proxy["http_proxy"] is None):
        proxy["http_proxy"] = os.getenv("http_proxy")
    if ("https_proxy" not in proxy) or (proxy["https_proxy"] is None):
        proxy["https_proxy"] = os.getenv("https_proxy")
    # delete all items in the dict whose value is None
    proxy = {k: v for k, v in proxy.items() if v is not None}
    logger.info("Setting proxy of OpenAI: %s", proxy)
    openai.proxy = proxy
    return proxy


@singleton
def init_openai(api_key: Optional[str] = None,
                use_proxy: Optional[bool] = None,
                proxy: Optional[Dict] = None) -> None:
    """
    Initializes the OpenAI APIs.

    :param api_key: the API KEY of the OpenAI. If it is None, the program will
     try to get it from the environment variable "OPENAI_API_KEY" or from the
     configuration file "~.openai/config".
    :param use_proxy: whether to use the proxy. If it is None, the program will
     set the proxy if "https://www.google.com" is not accessible.
    :param proxy: the proxy setting of the OpenAI.
    """
    logger.info("Initializing the OpenAI's API ...")
    set_api_key(api_key)
    if use_proxy is None:
        # Use proxy to bypass the China Greate Firewall
        logger.info("Testing the accessibility of https://www.google.com ...")
        accessible = is_website_accessible("https://www.google.com")
        logger.info("The website %s accessible.", "is" if accessible else "is NOT")
        use_proxy = not accessible
    if use_proxy is True:
        set_proxy(proxy)
    logger.info("The OpenAI's API was successfully initialized.")
