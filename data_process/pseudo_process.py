#encoding: utf-8
'''
@File    :   pseudo_process.py
@Time    :   2021/05/26 15:24:37
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''
import sys
sys.path.append("../../MTError")

import json
import random

import ahocorasick
from pypinyin import lazy_pinyin

from util.file_util import FileUtil
from util.log_util import LogUtil

class PseudoProcess(object):
    """
    伪训练预料构建
    """
    def generate_replace_error(self, train_data_path, error_map_path, train_replace_error_path):
        """
        检测能生成替换错误的位置
        @param:
        @return:
        """
        train_text_list = FileUtil.read_raw_data(train_data_path)
        error_map_dict = {item.split("\t")[0]: json.loads(item.split("\t")[1]) for item in FileUtil.read_raw_data(error_map_path)}

        # 打乱数据
        index_list = list(range(len(train_text_list)))
        random.shuffle(index_list)

        # 每个错误词对应的句子数量
        error_word_num_dict = {}

        # 构建词语ac自动机(先去除单字)
        word_trie = ahocorasick.AhoCorasick([word for word in error_map_dict.keys() if len(word) > 1]) 
        # 构建单字ac自动机
        char_trie = ahocorasick.AhoCorasick([word for word in error_map_dict.keys() if len(word) == 1]) 
        
        # 检测能够替换的词及位置
        replace_word_obj_list = []
        replace_char_obj_list = []
        num = 0
        for index in index_list:
            text = train_text_list[index]
            # 寻找多字错误
            word_result_set = word_trie.search(text, with_index=True)
            
            wrong_list = []
            for word, (start_index, end_index) in word_result_set:
                error_word_num_dict[word] = error_word_num_dict.get(word, 0) + 1
                # 某个词的错误数量达到设定阈值后则不再加入
                if error_word_num_dict[word] < 400:
                    wrong_list.append([word, start_index, end_index])

            if len(wrong_list) > 0:
                replace_word_obj_list.append({
                    "index": index,
                    "text": text,
                    "errors": wrong_list
                })
            else:
                # 当句子中不包含词语错误时，寻找单字错误
                char_result_set = char_trie.search(text, with_index=True)
                for single_word, (start_index, end_index) in char_result_set:
                    error_word_num_dict[single_word] = error_word_num_dict.get(single_word, 0) + 1
                    # 某个字的错误数量达到设定阈值后则不再加入
                    if error_word_num_dict[single_word] < 1600:
                        wrong_list.append([single_word, start_index, end_index])
                
                if len(wrong_list) > 0:
                    replace_char_obj_list.append({
                        "index": index,
                        "text": text,
                        "errors": wrong_list
                    })
            num += 1
            if num % 100000 == 0:
                LogUtil.logger.info(num)

        error_num = len(replace_word_obj_list) + len(replace_char_obj_list)
        LogUtil.logger.info("定位错误句子数量: {0}".format(error_num))

        replace_obj_list = replace_word_obj_list+replace_char_obj_list
        for replace_error_obj in replace_obj_list:
            # 每个句子最多保持1个错误，将多的错误移除
            if len(replace_error_obj["errors"]) > 1:
                replace_error_obj["errors"] = random.sample(replace_error_obj["errors"], 1)

            # 按比例从候选中选择错误词语
            for item in replace_error_obj["errors"]:
                replace_word_dict = error_map_dict.get(item[0], {})
                if len(replace_word_dict) > 0:
                    replace_word_list = [word for word, num in replace_word_dict.items()]
                    word_weight_list = [num/sum(replace_word_dict.values()) for word, num in replace_word_dict.items()]
                    replace_word = random.choices(replace_word_list, word_weight_list, k=1)[0]
                    item.append(replace_word)
        
        # 将替换错误结果存储
        FileUtil.write_json_data(replace_obj_list, train_replace_error_path)
    
    def generate_insert_delete_error(self, train_data_path, word_co_occurrence_path, train_replace_error_path, train_insert_delete_error_path):
        """
        生成插入删除错误
        @param:
        @return:
        """
        train_text_list = FileUtil.read_raw_data(train_data_path)
        word_co_occurrence_dict = {ele.split("\t")[0]: json.loads(ele.split("\t")[1]) \
            for ele in FileUtil.read_raw_data(word_co_occurrence_path) if len(ele.split("\t")) == 2}
        replace_error_index_set = {ele["index"] for ele in FileUtil.read_json_data(train_replace_error_path)}

        # 选取替换的5%作为插入删除错误
        normal_index_set = set(range(len(train_text_list))) - replace_error_index_set
        insert_delete_index_set = set(random.sample(normal_index_set, int(len(replace_error_index_set) * 0.05)))
        insert_index_set = set(random.sample(insert_delete_index_set, int(len(insert_delete_index_set) * 0.4)))
        delete_index_set = insert_delete_index_set - insert_index_set

        insert_delete_obj_list = []
        for index, text in enumerate(train_text_list):
            if len(text) < 3:
                continue

            # 对正确句子进行插入，则对asr句子来说就是删除错误; 选取随机位置插入，同时选取经常共现的字进行插入
            if index in insert_index_set:
                insert_index = random.randint(0, len(text)-1)
                # 选取经常共现的字进行插入
                word = text[insert_index]
                if word in word_co_occurrence_dict:
                    co_occurrence_dict = word_co_occurrence_dict[word]
                    insert_weight_list = [num/sum(co_occurrence_dict.values()) for word, num in co_occurrence_dict.items()]
                    insert_word = random.choices(list(co_occurrence_dict.keys()), insert_weight_list, k=1)[0]
                    insert_delete_obj_list.append({
                        "index": index,
                        "text": text,
                        "errors": [[text[insert_index], insert_index, insert_index+1, word+insert_word]]
                    })

            # 对正确句子进行删除，对asr句子来说就是插入错误; 随机选取位置删除
            elif index in delete_index_set:
                text_len = len(text)
                delete_index = random.randint(0, text_len-2)
                insert_delete_obj_list.append({
                    "index": index,
                    "text": text,
                    "errors": [[text[delete_index:delete_index+2], delete_index, delete_index+2, text[delete_index+1:delete_index+2]]]
                })

        # 将错误结果存储
        FileUtil.write_json_data(insert_delete_obj_list, train_insert_delete_error_path)

    def generate_format_train_data(self, train_data_path, train_replace_error_path, \
        train_insert_delete_error_path, train_format_path, is_pipeline_detect=False, is_pipeline_correct=False):
        """
        生成格式化的训练数据
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
        choose_right_id_set = set(random.sample(right_id_set, int(len(right_id_set) * 0.1)))

        train_format_list = []
        error_num = 0
        for index, text in enumerate(train_text_list):
            if index not in error_id_dict:
                # 当构建pipeline模型的纠错数据时, 不加入正常句子
                if is_pipeline_correct or len(train_format_list) > 4000000:
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
                if error_word == "" or len("".join(error_pinyin_list)) + len(error_word) < len(source_word):
                    continue
                
                asr_text = label_text[:start_index] + error_word + label_text[end_index:]

                correct_input = label_text[:start_index]
                for char_index, error_char in enumerate(error_word):
                    correct_input += error_char + " " + " ".join(error_pinyin_list[char_index]) + " "
                correct_input += label_text[end_index:]

                # 获取label在加入拼音后对应的index
                if len(error_word) == len(source_word):
                    # 长度相同时则直接对应到错误字
                    correct_error_index = []
                    token_index = start_index
                    for char_pinyin in error_pinyin_list:
                        correct_error_index.append(token_index)
                        token_index += len(char_pinyin) * 2 + 2
                elif len(error_word) < len(source_word):
                    # 错误词语长度更短时，按顺序对齐
                    correct_error_index = [start_index+choose_index*2 for choose_index in range(len(source_word))]
                else:
                    # 错误词语长度更长时，按顺序对齐
                    correct_error_index = [start_index+choose_index*2 for choose_index in range(len(source_word))]

                    # 错误词语长度更长时，则label首位拼音跟error对齐
                    # error_first_char_list = [ele[0] for ele in error_pinyin.split(" ")]
                    # error_first_char_set = set(error_first_char_list)
                    # label_first_char_list = [ele[0] for ele in label_pinyin.split(" ")]
                    # diff_num = len(error_word) - len(source_word)
                    # label_index_list = []
                    # for label_i, label_char in enumerate(label_first_char_list):
                    #     margin = label_i + diff_num
                    #     if label_char in error_first_char_set:
                    #         label_index = max(label_i, min(error_first_char_list.index(label_char), margin))
                    #         label_index_list.append(label_index)
                    #     else:
                    #         if len(label_index_list) > 0:
                    #             # 在先前对齐基础上按最小顺序对齐
                    #             label_index_list.append(max(label_index_list)+1)
                    #         else:
                    #             # 按最小顺序对齐
                    #             label_index_list.append(label_i)
                    
                    # label_index_set = set(label_index_list)
                    # correct_error_index = []
                    # token_index = start_index + 1
                    # for ele_index, ele in enumerate(error_pinyin.split(" ")):
                    #     if ele_index in label_index_set:
                    #         correct_error_index.append(token_index)
                    #     token_index += len(ele) * 2

                # 此处start,end是相对于asr
                errors = [{
                    "error_word": error_word,
                    "label_word": source_word,
                    "error_pinyin": " ".join(error_pinyin_list), 
                    "label_pinyin": " ".join(label_pinyin_list), 
                    "detect_error_range": [start_index, start_index+len(error_word)],
                    "correct_error_range": [start_index, (len("".join(error_pinyin_list))+len(error_word))*2+start_index],
                    "correct_error_align_index": correct_error_index,
                    "correct_input": correct_input
                }]
                train_format_list.append({
                    "asr": asr_text,
                    "transcript": label_text,
                    "errors": errors
                })
                error_num += 1

            if index % 100000 == 0:
                LogUtil.logger.info(index)
            
            # if index > 10000:
            #     break
        
        LogUtil.logger.info("error num: {0}".format(error_num))
        FileUtil.write_json_data(train_format_list, train_format_path)

    
    
if __name__ == "__main__":
    train_data_path = "/ssd1/users/fangzheng/data/mt_error/source_data/pseudo_train/source_train/combine_train.txt"
    word_co_occurrence_path = "/ssd1/users/fangzheng/data/mt_error/source_data/dict/zh_cidian/word_co_occurrence.txt"
    error_map_path = "/ssd1/users/fangzheng/data/mt_error/source_data/dict/asr_magic_align_filter.txt"
    train_replace_error_joint_path = "/ssd1/users/fangzheng/data/mt_error/source_data/pseudo_train/replace_error/all_replace_error.txt"
    train_insert_delete_error_path = "/ssd1/users/fangzheng/data/mt_error/source_data/pseudo_train/train_insert_delete_error.txt"
    train_joint_format_path = "/ssd1/users/fangzheng/data/mt_error/correct_format/pseudo_joint_train_format_400.txt"
    train_same_length_path = "/ssd1/users/fangzheng/data/mt_error/correct_format/pseudo_train_same_length.txt"

    pseudo_process = PseudoProcess()
    
    # pseudo_process.generate_replace_error(train_data_path, error_map_path, train_replace_error_combine_path)
    # pseudo_process.generate_insert_delete_error(train_data_path, word_co_occurrence_path, train_replace_error_combine_path, train_insert_delete_error_path)
    # pseudo_process.generate_format_train_data(train_data_path, train_replace_error_joint_path, train_insert_delete_error_path, train_joint_format_path, is_pipeline_correct=False)
