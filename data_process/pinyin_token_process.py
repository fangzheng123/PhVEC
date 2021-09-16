#encoding: utf-8
'''
@File    :   pinyin_token_process.py
@Time    :   2021/08/24 19:45:59
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import sys
sys.path.append("../../MTError")

import re
from pypinyin import lazy_pinyin, pinyin, Style

from util.file_util import FileUtil

class PinyinTokenProcess(object):
    """
    生成Pinyin Token, 用于新增词表
    """
    def generate_all_pinyin(self, chinese_path, pinyin_path):
        """
        生成所有的拼音
        """
        raw_data_list = FileUtil.read_raw_data(chinese_path)

        pinyin_token_set = set()
        for item in raw_data_list:
            token = item.split("#")[-1].strip()
            pinyin_token = lazy_pinyin(token)[0]
            if pinyin_token == token or not re.search('[a-z]', pinyin_token) or len(pinyin_token) < 2:
                continue

            pinyin_token_set.add(pinyin_token)
        
        FileUtil.write_raw_data(list(pinyin_token_set), pinyin_path)
    
    def generate_all_initials_finals(self, chinese_path, initial_final_path):
        """
        生成所有的声母和韵母
        @param:
        @return:
        """
        raw_data_list = FileUtil.read_raw_data(chinese_path)

        pinyin_token_set = set()
        for item in raw_data_list:
            token = item.split("#")[-1].strip()
            initial_token = pinyin(token, style=Style.INITIALS)[0][0]
            final_token = pinyin(token, style=Style.FINALS)[0][0]
            if not (initial_token == token or not re.search('[a-z]', initial_token) or len(initial_token) < 2):
                pinyin_token_set.add(initial_token)
            if not (final_token == token or not re.search('[a-z]', final_token) or len(final_token) < 2):
                pinyin_token_set.add(final_token)
        
        FileUtil.write_raw_data(list(pinyin_token_set), initial_final_path)


if __name__ == "__main__":
    chinese_path = "/ssd1/users/fangzheng/data/mt_error/source_data/dict/zh_cidian/pinyin.txt"
    pinyin_path = "/ssd1/users/fangzheng/data/mt_error/source_data/dict/zh_cidian/pinyin_token.txt"
    initial_final_path = "/ssd1/users/fangzheng/data/mt_error/source_data/dict/zh_cidian/initial_final_token.txt"

    pinyin_token_process = PinyinTokenProcess()
    # pinyin_token_process.generate_all_pinyin(chinese_path, pinyin_path)
    pinyin_token_process.generate_all_initials_finals(chinese_path, initial_final_path)




