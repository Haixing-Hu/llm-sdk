# ==============================================================================
#                                                                              =
#    Copyright (c) 2023. Haixing Hu                                            =
#    All rights reserved.                                                      =
#                                                                              =
# ==============================================================================
import unittest

from llmsdk.llm.tokenizer import OpenAiTokenizer
from llmsdk.common import ChatMessage


class TestOpenAiTokenizer(unittest.TestCase):

    def test_count_tokens_text_davinci_002(self):
        tokenizer = OpenAiTokenizer("text-davinci-002")
        text = "hello world"
        result = tokenizer.count_tokens(text)
        self.assertEqual(result, 2)

        text = "\nHello there, how may I assist you today?"
        result = tokenizer.count_tokens(text)
        self.assertEqual(result, 11)

        text = "你好，世界！"
        result = tokenizer.count_tokens(text)
        self.assertEqual(result, 14)

    def test_count_tokens_gpt_3_5_turbo(self):
        tokenizer = OpenAiTokenizer("gpt-3.5-turbo")
        text = "你好，世界！"
        result = tokenizer.count_tokens(text)
        self.assertEqual(result, 7)

    def test_count_message_tokens(self):
        tokenizer = OpenAiTokenizer("gpt-3.5-turbo")
        messages = [ChatMessage("user", "Hello!")]
        result = tokenizer.count_message_tokens(messages)
        self.assertEqual(result, 10)


if __name__ == '__main__':
    unittest.main()
