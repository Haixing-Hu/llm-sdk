# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################
import unittest

from llmsdk.util.openai_utils import (
    get_model_tokens,
    get_chunked_tokens,
    set_openai_proxy,
)
from llmsdk.embedding.openai_embedding import OpenAiEmbedding
from llmsdk.llm.tokenizer import OpenAiTokenizer


class TestOpenAiUtil(unittest.TestCase):
    def test_get_model_tokens(self):
        self.assertEqual(get_model_tokens("text-davinci-002"), 4097)
        self.assertEqual(get_model_tokens("code-davinci-002"), 8001)
        self.assertEqual(get_model_tokens("gpt-3.5-turbo"), 4096)
        self.assertEqual(get_model_tokens("gpt-4"), 8192)

    def test_get_chunked_tokens(self):
        model = OpenAiEmbedding.DEFAULT_MODEL
        tokenizer = OpenAiTokenizer(model)
        text = "The food was delicious and the waiter..."
        result = get_chunked_tokens(model, tokenizer, text)
        expected = [[791, 3691, 574, 18406, 323, 279, 68269, 1131]]
        self.assertEqual(result, expected)

        text = "The food was delicious and the waiter..." * 3000
        tokens = [791, 3691, 574, 18406, 323, 279, 68269, 1131] * 3000
        chunk_size = 8191
        result = get_chunked_tokens(model, tokenizer, text)
        expected = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
        self.assertEqual(result, expected)

    def test_set_proxy(self):
        result = set_openai_proxy(None)
        print(result)
        # self.assertIsNotNone(result)
        # self.assertNotEqual({}, result)


if __name__ == '__main__':
    unittest.main()
