# ##############################################################################
#                                                                              #
#     Copyright (c) 2022 - 2023.                                               #
#     Haixing Hu, Qubit Co. Ltd.                                               #
#                                                                              #
#     All rights reserved.                                                     #
#                                                                              #
# ##############################################################################

CONFIG = {
    "prompt_template": {
        "instruction_template": "你的任务是根据给定的某个记录的字段，在给定的表格数据中"
                                "找到和该记录字段最匹配的1行数据，并直接输出该匹配行的指"
                                "定某列的值。\n"
                                "给定的表格数据包含在3个单引号中，以符合RFC-4180的CSV"
                                "格式表示，其中第一行表示列名称。\n"
                                "给定的记录，也包含在3个单引号中，每一行表示一个字段，"
                                "字段名和字段值之间用冒号隔开。\n"
                                "你的回复只能是最匹配的行的指定列的值。如果找不到最匹配的"
                                "行，只能输出一个字符串 'NONE'。",
        "example_input_prefix": "输入：\n",
        "example_output_prefix": "输出：\n",
    },
    "record_score_threshold": 0.85,
}