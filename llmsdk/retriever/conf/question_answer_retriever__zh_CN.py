# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################

CONFIG = {
    "unknown_question_answer": "很抱歉，我不知道此问题的答案。请联系人工客服。",
    "prompt_template": {
        "instruction_template": "请根据下面给出的信息回答最后给出的问题，你的回答必须专业、准确。"
                                "如果不知道答案，请回答“{unknown_question_answer}”，"
                                "不要试图捏造答案。",
        "prompt_template": "{question}",
        "example_input_prefix": "问题：",
        "example_output_prefix": "回答："
    },
    "direct_answer_score_threshold": 0.92,
    "question_score_threshold": 0.85,
    "answer_score_threshold": 0.75,
    "question_limit": 5,
    "answer_limit": 5,
    "history_limit": 5,
}
