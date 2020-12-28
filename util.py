# coding=utf-8
"""
工具封装
illool@163.com
QQ:122018919
"""


def q_to_b(q_str):
    """全角转半角"""
    b_str = ""
    for uchar in q_str:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif 65374 >= inside_code >= 65281:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        # Python2中使用的chr()将Ascii的值转换成对应字符，unichr()将Unicode的值转换成对应字符
        # b_str += unichr(inside_code)
        b_str += chr(inside_code)
    return b_str


def b_to_q(b_str):
    """半角转全角"""
    q_str = ""
    for uchar in b_str:
        inside_code = ord(uchar)
        if inside_code == 32:  # 半角空格直接转化
            inside_code = 12288
        elif 126 >= inside_code >= 32:  # 半角字符（除空格）根据关系转化
            inside_code += 65248
        # q_str += unichr(inside_code)
        q_str += chr(inside_code)
    return q_str
