# ##############################################################################
#                                                                              #
#    Copyright (c) 2023.  Haixing Hu.                                          #
#                                                                              #
#    All rights reserved.                                                      #
#                                                                              #
# ##############################################################################
import logging
import unittest

from llmdk.util.openai_utils import (
    get_model_tokens,
    count_tokens,
    count_message_tokens,
    get_chunked_tokens,
    set_proxy,
)
from llmdk.embedding.openai_embedding import DEFAULT_MODEL as DEFAULT_EMBEDDING_MODEL


class TestOpenAiUtil(unittest.TestCase):
    def test_get_model_tokens(self):
        self.assertEqual(get_model_tokens("text-davinci-002"), 4097)
        self.assertEqual(get_model_tokens("code-davinci-002"), 8001)
        self.assertEqual(get_model_tokens("gpt-3.5-turbo"), 4096)
        self.assertEqual(get_model_tokens("gpt-4"), 8192)

    def test_count_tokens(self):
        model = "text-davinci-002"
        text = "hello world"
        result = count_tokens(model, text)
        self.assertEqual(result, 2)
        text = "\nHello there, how may I assist you today?"
        result = count_tokens(model, text)
        self.assertEqual(result, 11)
        text = "你好，世界！"
        result = count_tokens(model, text)
        self.assertEqual(result, 14)
        model = "gpt-3.5-turbo"
        text = "你好，世界！"
        result = count_tokens(model, text)
        self.assertEqual(result, 7)

    def test_count_message_tokens(self):
        model = "gpt-3.5-turbo"
        logger = logging.getLogger("TestOpenAiUtil")
        messages = [{"role": "user", "content": "Hello!"}]
        result = count_message_tokens(model, messages, logger)
        self.assertEqual(result, 10)

    def test_get_chunked_tokens(self):
        model = DEFAULT_EMBEDDING_MODEL
        text = "The food was delicious and the waiter..."
        result = get_chunked_tokens(model, text)
        expected = [[791, 3691, 574, 18406, 323, 279, 68269, 1131]]
        self.assertEqual(result, expected)

        text = "The food was delicious and the waiter..." * 3000
        tokens = [791, 3691, 574, 18406, 323, 279, 68269, 1131] * 3000
        chunk_size = 8191
        result = get_chunked_tokens(model, text)
        expected = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
        self.assertEqual(result, expected)

    def test_set_proxy(self):
        result = set_proxy(None)
        print(result)
        # self.assertIsNotNone(result)
        # self.assertNotEqual({}, result)


if __name__ == '__main__':
    unittest.main()
