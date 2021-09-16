#encoding: utf-8
'''
@File    :   text_util.py
@Time    :   2021/08/19 10:33:54
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''
import re

class TextUtil(object):
    """
    文本处理工具类
    """
    @classmethod
    def filter_symbol(cls, text):
        """
        过滤标点符号
        @param:
        @return:
        """
        new_text = re.sub("[、,.，。？！!\-《》：“”\'\"]", "", text.strip())
        return new_text