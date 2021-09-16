#encoding: utf-8
'''
@File    :   pseudo_pinyin_token_format.py
@Time    :   2021/08/24 21:13:28
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import sys
sys.path.append("../../MTError")

import re
import json
import random
import multiprocessing

import ahocorasick
from pypinyin import lazy_pinyin

from util.file_util import FileUtil
from util.log_util import LogUtil

PINYIN_PAD = "[UNK]"

class PseudoPinyinTokenFormat(object):
    """
    伪训练语料格式化
    """

    def generate_complete_token_train_data(self, train_data_path, train_replace_error_path, train_format_path):
        """
        新增完整token
        @param:
        @return:
        """
        train_text_list = FileUtil.read_raw_data(train_data_path)
        replace_error_obj_list = FileUtil.read_json_data(train_replace_error_path)
        error_obj_list = replace_error_obj_list
        error_id_dict = {ele["index"]: ele for ele in error_obj_list}

        # 随机选择正例
        right_id_set = set(range(len(train_text_list))) - set(error_id_dict.keys())
        choose_right_id_set = set(random.sample(right_id_set, int(len(right_id_set) * 0.05)))

        train_format_list = []
        error_num = 0
        for index, text in enumerate(train_text_list):
            if index not in error_id_dict:
                if len(train_format_list) > 1800000:
                    continue
                
                if index in choose_right_id_set:
                    train_format_list.append({
                        "asr": text,
                        "transcript": text,
                        "errors": []
                    })
            else:
                error_obj = error_id_dict[index]
                label_text = error_obj["text"]
                source_word, start_index, end_index, error_word = error_obj["errors"][0]
                error_pinyin_list = lazy_pinyin(error_word)
                label_pinyin_list = lazy_pinyin(source_word)
                
                if len(source_word) != len(error_word):
                    continue
                
                asr_text = label_text[:start_index] + error_word + label_text[end_index:]

                correct_input_text = label_text[:start_index]
                for char_index, error_char in enumerate(error_word):
                    correct_input_text += error_char + " " + error_pinyin_list[char_index] + " "
                correct_input_text += label_text[end_index:]

                correct_label_text = label_text[:start_index]
                for char_index, source_char in enumerate(source_word):
                    correct_label_text += source_char + " " + source_word[char_index] + " "
                correct_label_text += label_text[end_index:]

                # 此处start,end是相对于asr
                errors = [{
                    "error_word": error_word,
                    "label_word": source_word,
                    "error_pinyin": " ".join(error_pinyin_list), 
                    "label_pinyin": " ".join(label_pinyin_list), 
                    "detect_error_range": [start_index, start_index+len(error_word)],
                    "correct_input": correct_input_text,
                    "correct_label": correct_label_text
                }]
                train_format_list.append({
                    "asr": asr_text,
                    "transcript": label_text,
                    "errors": errors
                })
                error_num += 1

            if index % 100000 == 0:
                LogUtil.logger.info(index)
            
            # if index > 1000:
            #     break
        
        LogUtil.logger.info("error num: {0}".format(error_num))
        FileUtil.write_json_data(train_format_list, train_format_path)

    def filter_en_data(self, train_format_path, filter_format_path):
        """
        过滤包含英文字母的数据
        @param:
        @return:
        """
        all_data_obj_list = FileUtil.read_json_data(train_format_path)

        filter_obj_list = []
        for data_obj in all_data_obj_list:
            if re.search('[a-zA-Z]', data_obj["asr"]) or re.search('[a-zA-Z]', data_obj["transcript"]):
                continue
            
            filter_obj_list.append(data_obj)
        
        LogUtil.logger.info("剩余数量: {0}, 过滤数量: {1}".format(len(filter_obj_list), len(all_data_obj_list)-len(filter_obj_list)))
        FileUtil.write_json_data(filter_obj_list, filter_format_path)

if __name__ == "__main__":
    train_data_path = "/ssd1/users/fangzheng/data/mt_error/source_data/pseudo_train/source_train/combine_train.txt"
    train_replace_error_joint_path = "/ssd1/users/fangzheng/data/mt_error/source_data/pseudo_train/replace_error/all_replace_error.txt"
    train_joint_format_path = "/ssd1/users/fangzheng/data/mt_error/compare_format/pseudo_joint_train_pinyin_format.txt"
    filter_format_path = "/ssd1/users/fangzheng/data/mt_error/compare_format/pseudo_joint_train_pinyin_filter_format.txt"

    pseudo_format = PseudoPinyinTokenFormat()
    # pseudo_format.generate_complete_token_train_data(train_data_path, train_replace_error_joint_path, train_joint_format_path)   
    pseudo_format.filter_en_data(train_joint_format_path, filter_format_path)