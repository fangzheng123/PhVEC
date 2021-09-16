#encoding: utf-8
'''
@File    :   error_process.py
@Time    :   2021/05/25 20:28:16
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''
import sys
sys.path.append("../../MTError")

import re
import json
from collections import Counter
from difflib import SequenceMatcher

import Levenshtein
from pypinyin import lazy_pinyin

from util.asr_score_util import ASRScoreUtil
from util.file_util import FileUtil
from util.log_util import LogUtil
from util.text_util import TextUtil

class ErrorProcess(object):
    """
    ASR错误处理
    """
    def filter_asr_data(self, source_asr_path, filter_asr_path):
        """
        过滤部分错误率较大的ASR数据
        @param:
        @return:
        """
        data_obj_list = FileUtil.read_json_data(source_asr_path)

        filter_obj_list = []
        for data_obj in data_obj_list:
            asr_sent = data_obj["asr"]
            transcript = data_obj["transcript"]

            # 错误字数超过2个则去除
            if abs(len(asr_sent)-len(transcript)) > 2 or Levenshtein.ratio(asr_sent, transcript) < 0.7:
                print(asr_sent, transcript)
                continue
                
            filter_obj_list.append(data_obj)
        
        print("source num: {0}, filter num: {1}".format(len(data_obj_list), len(filter_obj_list)))
        cer = ASRScoreUtil.calculate_cer([ele["asr"] for ele in filter_obj_list], [ele["transcript"] for ele in filter_obj_list])
        print("CER is {0:.2f}%".format(cer*100))

        FileUtil.write_json_data(filter_obj_list, filter_asr_path)

    def asr_error_analyse(self, format_data_path, error_align_path, source_type):
        """
        分析asr识别错误类型, 并将错误对存储到文件中
        @param:
        @return:
        """
        data_obj_list = FileUtil.read_json_data(format_data_path)

        phonetic_diff_num = 0
        phonetic_same_num = 0
        error_sent_num = 0
        error_num_dict = {}
        tag_num_dict = {}
        error_word_dict = {}
        replace_num_dict = {}
        replace_len_dict = {}
        for data_obj in data_obj_list:
            asr_sent = data_obj["asr"]
            transcript = data_obj["transcript"]

            if asr_sent != transcript:
                error_sent_num += 1

            if asr_sent != transcript:
                diff_len = len(asr_sent)-len(transcript)
                error_num_dict[diff_len] = error_num_dict.get(diff_len, 0) + 1
                s = SequenceMatcher(None, asr_sent, transcript)
                    
                # print(asr_sent, "####", transcript)
                for tag, i1, i2, j1, j2 in s.get_opcodes():
                    if tag != "equal":
                        asr_word, transcript_word = asr_sent[i1:i2], transcript[j1:j2]
                        asr_pinyin, transcript_pinyin = "".join(lazy_pinyin(asr_word)), "".join(lazy_pinyin(transcript_word))
                        if asr_pinyin != transcript_pinyin:
                            phonetic_diff_num += 1
                        else:
                            phonetic_same_num += 1
                        
                        tag_num_dict[tag] = tag_num_dict.get(tag, 0) + 1
                        if tag == "replace":
                            error_word_dict.setdefault(transcript_word, []).append(asr_word)
                            
                            replace_diff = len(asr_word) - len(transcript_word)
                            replace_num_dict[replace_diff] = replace_num_dict.get(replace_diff, 0) + 1
                            replace_len_dict[len(transcript_word)] = replace_len_dict.get(len(transcript_word), 0) + 1
                        
                        # print(tag, asr_word, transcript_word, asr_pinyin, transcript_pinyin)
                # print("")
        
        print("replace diff len ratio:", {diff_len: num/sum(replace_num_dict.values()) for diff_len, num in replace_num_dict.items()})
        print("replace len ratio:", {word_len: num/sum(replace_len_dict.values()) for word_len, num in replace_len_dict.items()})
        print({tag: num/sum(tag_num_dict.values()) for tag, num in tag_num_dict.items()})
        print("pinyin same ratio:", phonetic_same_num/(phonetic_same_num+phonetic_diff_num))
        print({k: v/sum(error_num_dict.values()) for k, v in error_num_dict.items()})
        print("error sent num: {0}, error sent ratio: {1}".format(error_sent_num, error_sent_num/len(data_obj_list)))
        # 计算CER
        cer = ASRScoreUtil.calculate_cer([ele["asr"] for ele in data_obj_list], [ele["transcript"] for ele in data_obj_list])
        print("CER is {0:.2f}%".format(cer*100))

        # 存储错误平行对
        # error_word_align_list = [{"label": label, "confusion": error_list, "source": source_type} for label, error_list in error_word_dict.items()]
        # FileUtil.write_json_in_append(error_word_align_list, error_align_path)
    
    def filter_symbol(self, text):
        """
        过滤标点符号
        @param:
        @return:
        """
        new_text = re.sub("[、,.，。？！!\-《》：“”\'\"]", "", text.strip())
        return new_text

    def filter_error_pair(self, error_align_path, error_filter_path):
        """
        过滤部分结果对
        @param:
        @return:
        """
        error_obj_list = FileUtil.read_json_data(error_align_path)

        # 合并相同label
        filter_label_dict = {}
        for error_obj in error_obj_list:
            label = error_obj["label"]
            label = self.filter_symbol(label)
            # 忽略长度大于4的词
            if len(label) > 4 or len(label) == 0:
                continue
                        
            label_pinyin = "".join(lazy_pinyin(label))
            confusion_list = error_obj["confusion"]
            word_counter = Counter(confusion_list)
            filter_word_list = []
            for word, num in word_counter.items():
                word = self.filter_symbol(word)
                if len(word) == 0 or abs(len(word)-len(label)) > 2 or word.lower() == label.lower():
                    continue
                word_pinyin = "".join(lazy_pinyin(word))
                if error_obj["source"] in ["magic", "aishell"]:
                    filter_word_list.extend([word] * num)
                elif Levenshtein.ratio(label_pinyin, word_pinyin) > 0.5 or num > 0:
                    filter_word_list.extend([word] * num)
            
            filter_label_dict.setdefault(label, []).extend(filter_word_list)
        
        filter_label_list = [label + "\t" + json.dumps(dict(Counter(word_list)), ensure_ascii=False) \
             for label, word_list in filter_label_dict.items() if len(word_list) > 0]
        FileUtil.write_raw_data(filter_label_list, error_filter_path)

if __name__ == "__main__":
    error_process = ErrorProcess()

    data_name = "bstc"
    base_dir = "/ssd1/users/fangzheng/data/mt_error/"
    format_data_path = base_dir + "asr_format/" + data_name + "_format/train_single_format.txt"
    # format_filter_path = base_dir + "asr_format/" + data_name + "_format/test_single_format_filter.txt"
    
    error_align_path = "/ssd1/users/fangzheng/data/mt_error/source_data/dict/asr_error_align_bstc.txt"
    error_filter_path = "/ssd1/users/fangzheng/data/mt_error/source_data/dict/asr_error_align_bstc_filter.txt"

    # 过滤部分magic的数据
    # error_process.filter_asr_data(format_data_path, format_filter_path)
    error_process.asr_error_analyse(format_data_path, error_align_path, data_name)
    # error_process.filter_error_pair(error_align_path, error_filter_path)

