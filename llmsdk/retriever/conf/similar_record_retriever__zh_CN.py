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
                                "找到和该记录最匹配的1行数据，并输出匹配行的指定列的值。\n"
                                "给定的表格数据和记录数据均包含在3个单引号中，"
                                "以符合RFC-4180的CSV格式表示，其中第一行表示列名称。\n"
                                "请一步一步思考。",
        "output_requirement_template": "你的回复必须是个JSON对象，包含\"explanation\""
                                       "和\"answer\"两个属性。其中， \"explanation\""
                                       "属性包含对\"answer\"属性值的解释，包含了推导出"
                                       "\"answer\"值的一步步思考过程；\"answer\""
                                       "属性值是找到的最匹配行的指定列的值; 如果找不到最"
                                       "匹配的行，或者不止一个最匹配的行，该属性值为字符"
                                       "串\"NONE\"。",
        "input_template": "表格数据：\n"
                          "'''\n"
                          "{known_records}"
                          "'''\n"
                          "记录数据：\n"
                          "'''\n"
                          "{query_record}"
                          "'''\n"
                          "输出列：'{id_field}'",
        "explanation_instruction": "请解释下最后的回答。",
        "example_input_prefix": "输入：\n",
        "example_input_suffix": "\n",
        "example_output_prefix": "输出：\n",
        "example_output_suffix": "\n\n",
        "examples": [{
            "input": "表格数据：\n"
                     "'''\n"
                     "编码,名称,规格\n"
                     "0084,负压采血管,EDTAK2 9ml\n"
                     "5237,大山楂颗粒,每袋装10g(低糖型)\n"
                     "4380,大山楂丸,每丸重9g\n"
                     "5903,九制大黄丸,每50粒重3g\n"
                     "'''\n"
                     "记录数据：\n"
                     "'''\n"
                     "名称\n"
                     "山楂丸\n"
                     "'''\n"
                     "输出列：'名称'",
            "output": "{"
                      "\"explanation\": \"待匹配记录'项目名称'为'山楂丸'，"
                      "在给定的表格数据中，没有'项目名称'列，但有'名称'列；"
                      "第2行的'名称'为'大山楂颗粒'，可以匹配'山楂丸'；"
                      "第3行的'名称'为'大山楂丸'，也可以匹配'山楂丸'；"
                      "但第3行的'名称'比第2行的'名称'更匹配'山楂丸'；"
                      "所以第3行为最匹配的行。"
                      "要求输出最匹配的行的'名称'列，所以输出的结果中'answer'为'大山楂丸'。\", "
                      "\"answer\": \"大山楂丸\""
                      "}",
        }, {
            "input": "表格数据：\n"
                     "'''\n"
                     "编码,名称,规格\n"
                     "0084,负压采血管,EDTAK2 9ml\n"
                     "5237,大山楂颗粒,每袋装10g(低糖型)\n"
                     "4380,大山楂丸,每丸重9g\n"
                     "5903,九制大黄丸,每50粒重3g\n"
                     "'''\n"
                     "记录数据：\n"
                     "'''\n"
                     "名称\n"
                     "地黄丸\n"
                     "'''\n"
                     "输出列：'编码'",
            "output": "{"
                      "\"explanation\": \"待匹配记录'项目名称'为'地黄丸'，"
                      "在给定的表格数据中，没有'项目名称'列，但有'名称'列；"
                      "第3行的'名称'为'大山楂颗粒'，和'地黄丸'有一个字'丸'匹配，但他们明显是不同的药品；"
                      "第4行的'名称'为'九制大黄丸'，也和'地黄丸'有一个字'丸'匹配，但他们也明显是不同的药品；"
                      "所以找不到最匹配的行，因此输出的结果中'answer'为'NONE'。\", "
                      "\"answer\": \"NONE\""
                      "}",
        }, {
            "input": "表格数据：\n"
                     "'''\n"
                     "编码,名称,规格,中文通用名\n"
                     "0084,负压采血管,EDTAK2 9ml,203-采血管\n"
                     "5237,大山楂颗粒,每袋装10g(低糖型),统筹外中成药\n"
                     "4380,大山楂丸,每丸重9g,统筹外中成药\n"
                     "5903,九制大黄丸,每50粒重3g,统筹外中成药\n"
                     "'''\n"
                     "记录数据：\n"
                     "'''\n"
                     "名称\n"
                     "中成药\n"
                     "'''\n"
                     "输出列：'名称'",
            "output": "{"
                      "\"explanation\": \"待匹配记录'名称'为'中成药'，在给定的表格数据中，"
                      "没有哪一行的'名称'可以匹配'中成药'；"
                      "但第1行和第2行的'中文通用名'都是'统筹外中成药'，都可以匹配'中成药'，且匹配程度相同；"
                      "因此找到2个最匹配的行；"
                      "按照要求，不止一条最匹配行，所以输出的结果中'answer'为'NONE'。\", "
                      "\"answer\": \"NONE\"}",
        # }, {
        #     "input": "表格数据：\n"
        #              "'''\n"
        #              "编码,名称,规格,中文通用名\n"
        #              "0074,负压采血管,促凝管+分离胶 1ml,203-采血管\n"
        #              "0084,负压采血管,EDTAK2 9ml,203-采血管\n"
        #              "5237,大山楂颗粒,每袋装10g(低糖型),统筹外中成药\n"
        #              "4380,大山楂丸,每丸重9g,统筹外中成药\n"
        #              "5903,九制大黄丸,每50粒重3g,统筹外中成药\n"
        #              "'''\n"
        #              "记录数据：\n"
        #              "'''\n"
        #              "名称,规格\n"
        #              "采血管,1ml\n"
        #              "'''\n"
        #              "输出列：'编码'",
        #     "output": "0074",
        }],
    },
    "record_limit": 10,
    "record_score_threshold": 0.85,
}
