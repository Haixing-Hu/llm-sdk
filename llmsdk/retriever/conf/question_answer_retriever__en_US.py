
# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################

CONFIG = {
    "unknown_question_answer": "Sorry, I don't know how to answer this question. "
                               "Please contact the customer service.",
    "prompt_template": {
        "instruction_template": "Use the following examples to answer the question at "
                                "the end. If you don't know the answer, just say "
                                "\"{unknown_question_answer}\", don't try to make up "
                                "an answer.",
        "prompt_template": "{question}",
        "example_input_prefix": "question: ",
        "example_output_prefix": "answer: ",
    },
    "direct_answer_score_threshold": 0.92,
    "question_score_threshold": 0.85,
    "answer_score_threshold": 0.75,
    "question_limit": 5,
    "answer_limit": 5,
    "history_limit": 5,
}
