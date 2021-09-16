#encoding: utf-8
'''
@File    :   pseudo_error_align.py
@Time    :   2021/06/22 20:10:58
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import sys
sys.path.append("../../MTError")

import json
import random
import multiprocessing

import ahocorasick
from pypinyin import lazy_pinyin

from util.file_util import FileUtil
from util.log_util import LogUtil

PINYIN_PAD = "[UNK]"

class PseudoDataFormat(object):
    """
    伪训练语料格式化
    """
    def align_pinyin(self, error_word, error_pinyin_list, label_word, label_pinyin_list):
        """
        对齐拼音
        @param:
        @return:
        """
        error_pinyin = "".join(error_pinyin_list)
        label_pinyin = "".join(label_pinyin_list)
        error_len = len(error_pinyin)
        label_len = len(label_pinyin)
        res = [["" for i in range(label_len + 1)] for j in range(error_len + 1)]
        
        # 动态规划获取最大公共子序列
        for i in range(1, error_len+1):
            for j in range(1, label_len + 1):
                if error_pinyin[i - 1] == label_pinyin[j - 1]:
                    res[i][j] = res[i - 1][j - 1] + error_pinyin[i - 1]
                else:
                    res[i][j] = res[i - 1][j] if len(res[i - 1][j]) > len(res[i][j - 1]) else res[i][j - 1]
        
        # 获取每个公共字符在两个拼音文本中的具体匹配位置
        result_index_dict = {}
        for i in range(error_len, 0, -1):
            j = sorted([(j, len(ele)) for j, ele in enumerate(res[i])], key=lambda x: x[1], reverse=True)[0][0]
            result_index_dict[res[i][j]] = (i-1, j-1)
        
        # 将label index与对应汉字对齐
        word_pinyin_num = 0
        label_index_word_dict = {}
        for word, label_pinyin in zip(label_word, label_pinyin_list):
            for i in range(len(label_pinyin)):
                label_index_word_dict[word_pinyin_num+i] = word
            word_pinyin_num += len(label_pinyin)
        
        # 将label汉字与error拼音对齐
        res_txt = res[-1][-1]
        align_dict = {}
        for i in range(len(res_txt), 0, -1):
            prefix = res_txt[:i]
            error_i, label_j = result_index_dict.get(prefix, (0, 0))
            align_dict[error_i] = label_index_word_dict.get(label_j, "")
        
        # 剩余未对齐位置填充PAD
        for error_pinyin in error_pinyin_list:
            for i in range(len(error_pinyin)):
                align_dict.setdefault(i, PINYIN_PAD)

        # 变长文本则需要将汉字和拼音同时对齐
        if len(error_word) != len(label_word):
            # error汉字对齐到label汉字
            word_pinyin_num = 0
            error_label_align_dict = {}
            for error_char, error_pinyin in zip(error_word, error_pinyin_list):
                for i in range(len(error_pinyin)):
                    if align_dict.get(word_pinyin_num+i, PINYIN_PAD) != PINYIN_PAD:
                        error_label_align_dict[error_char] = align_dict.get(word_pinyin_num+i, PINYIN_PAD)
                        break
                word_pinyin_num += len(error_pinyin)
            
            # 构建汉字+拼音对齐结果
            word_pinyin_num = 0
            align_label_list = []
            for error_char, error_pinyin in zip(error_word, error_pinyin_list):
                align_label_list.append(error_label_align_dict.get(error_char, PINYIN_PAD))
                align_label_list.extend([align_dict.get(word_pinyin_num+i, PINYIN_PAD) for i in range(len(error_pinyin))])
                word_pinyin_num += len(error_pinyin)

            # 如果label均未对应上，即训练将毫无收益，则直接按字等长对齐
            align_label_set = set(align_label_list)
            if len(align_label_set) == 1 and list(align_label_set)[0] == PINYIN_PAD:
                align_label_list = []
                for error_index, error_char in enumerate(error_word):
                    if error_index < len(label_word):
                        label_char = label_word[error_index]
                    else:
                        label_char = PINYIN_PAD
                    align_label_list.append(label_char)

                    error_pinyin = error_pinyin_list[error_index]
                    align_label_list.extend([PINYIN_PAD for i in range(len(error_pinyin))])
        
        # 定长文本直接返回拼音对齐结果即可
        else:
            align_label_list = [ele[1] for ele in sorted(align_dict.items(), key=lambda x: x[0])]
        
        return align_label_list
    
    def get_correct_label_list(self, error_word, error_pinyin_list, label_word, label_pinyin_list):
        """
        获得拼音标签
        @param:
        @return:
        """
        correct_label_list = []
        # 长度相同时按序对齐
        if len(error_word) == len(label_word):
            for label_index, label_char in enumerate(label_word):
                # 汉字按序对齐
                correct_label_list.append(label_char)
                # 只对齐单个字的拼音
                align_label_list = self.align_pinyin(error_word[label_index], \
                    [error_pinyin_list[label_index]], label_char, [label_pinyin_list[label_index]])
                correct_label_list.extend(align_label_list)
        
        # 长度不同时根据公共拼音对齐
        else:
            # 将汉字和拼音均打标，此时将所有拼音拼接后对齐
            align_label_list = self.align_pinyin(error_word, error_pinyin_list, label_word, label_pinyin_list)
            correct_label_list.extend(align_label_list)

        return correct_label_list
    
    def get_filter_correct_label_list(self, error_word, error_pinyin_list, label_word, label_pinyin_list):
        """
        获得拼音标签, 并对重复标签进行过滤
        @param:
        @return:
        """
        correct_label_list = []
        # 长度相同时按序对齐
        if len(error_word) == len(label_word):
            for label_index, label_char in enumerate(label_word):
                # 汉字按序对齐
                correct_label_list.append(label_char)
                # 拼音对应为[UNK]
                correct_label_list.extend([PINYIN_PAD] * len(error_pinyin_list[label_index]))
        
        # 长度不同时根据公共拼音对齐
        else:
            # 将汉字和拼音均打标，此时将所有拼音拼接后对齐
            align_label_list = self.align_pinyin(error_word, error_pinyin_list, label_word, label_pinyin_list)
            # 对重复标签进行过滤
            previous_set = set()
            filter_align_label_list = []
            for label_ele in align_label_list:
                if label_ele not in previous_set:
                    filter_align_label_list.append(label_ele)
                else:
                    filter_align_label_list.append(PINYIN_PAD)
                previous_set.add(label_ele)

            correct_label_list.extend(filter_align_label_list)

        return correct_label_list

    def generate_align_train_data(self, train_data_path, train_replace_error_path, \
        train_insert_delete_error_path, train_format_path, is_pipeline_correct=False):
        """
        生成拼音对齐的训练数据
        @param:
        @return:
        """
        train_text_list = FileUtil.read_raw_data(train_data_path)
        replace_error_obj_list = FileUtil.read_json_data(train_replace_error_path)
        # insert_delete_obj_list = FileUtil.read_json_data(train_insert_delete_error_path)
        # error_obj_list = replace_error_obj_list + insert_delete_obj_list
        error_obj_list = replace_error_obj_list
        error_id_dict = {ele["index"]: ele for ele in error_obj_list}

        # 随机选择正例
        right_id_set = set(range(len(train_text_list))) - set(error_id_dict.keys())
        choose_right_id_set = set(random.sample(right_id_set, int(len(right_id_set) * 0.05)))

        train_format_list = []
        error_num = 0
        for index, text in enumerate(train_text_list):
            if index not in error_id_dict:
                # 当构建pipeline模型的纠错数据时, 不加入正常句子
                if is_pipeline_correct or len(train_format_list) > 1800000:
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
                
                # 排除错误词为空的情况，即AB-{A,B,D}均可，但不允许AB映射到空字符(此时应该替换为AB周围词映射，例如CAB-C)
                # 词语对应的拼音比目标词更短时，无法映射
                if error_word == "" or len("".join(error_pinyin_list)) + len(error_word) < len(source_word) or len(error_word) > len(error_pinyin_list):
                    continue
                
                asr_text = label_text[:start_index] + error_word + label_text[end_index:]

                correct_input_text = label_text[:start_index]
                for char_index, error_char in enumerate(error_word):
                    correct_input_text += error_char + " " + " ".join(error_pinyin_list[char_index]) + " "
                correct_input_text += label_text[end_index:]

                correct_label_list = self.get_filter_correct_label_list(error_word, error_pinyin_list, source_word, label_pinyin_list)
                # correct_label_list = self.get_correct_label_list(error_word, error_pinyin_list, source_word, label_pinyin_list)
                correct_label_text = label_text[:start_index] + " ".join(correct_label_list) + " "
                correct_label_text += label_text[end_index:]
                # if len(source_word) != len(error_word):
                #     print(error_word, source_word, correct_input_text, correct_label_text)

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

    def filter_not_same_length_data(self, train_format_path, same_length_format_path):
        """
        过滤长度不一致的训练语料
        @param:
        @return:
        """
        all_data_obj_list = FileUtil.read_json_data(train_format_path)

        same_length_obj_list = []
        for data_obj in all_data_obj_list:
            if len(data_obj["asr"]) == len(data_obj["transcript"]):
                same_length_obj_list.append(data_obj)
        
        LogUtil.logger.info("相同长度数量: {0}, 过滤数量: {1}".format(len(same_length_obj_list), len(all_data_obj_list)-len(same_length_obj_list)))
        FileUtil.write_json_data(same_length_obj_list, same_length_format_path)

if __name__ == "__main__":
    train_data_path = "/ssd1/users/fangzheng/data/mt_error/source_data/pseudo_train/source_train/combine_train.txt"
    train_replace_error_joint_path = "/ssd1/users/fangzheng/data/mt_error/source_data/pseudo_train/replace_error/all_replace_error.txt"
    train_insert_delete_error_path = "/ssd1/users/fangzheng/data/mt_error/source_data/pseudo_train/train_insert_delete_error.txt"
    train_joint_format_path = "/ssd1/users/fangzheng/data/mt_error/correct_format/pseudo_joint_train_pinyin_format.txt"
    train_same_length_path = "/ssd1/users/fangzheng/data/mt_error/correct_format/pseudo_train_same_length.txt"

    pseudo_format = PseudoDataFormat()
    pseudo_format.generate_align_train_data(train_data_path, train_replace_error_joint_path, train_insert_delete_error_path, train_joint_format_path)   
    # pseudo_format.filter_not_same_length_data(train_joint_format_path, train_same_length_path)