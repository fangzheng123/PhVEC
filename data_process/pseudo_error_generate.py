#encoding: utf-8
'''
@File    :   pseudo_error_generate.py
@Time    :   2021/06/21 16:36:36
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

def replace_word_task(train_text_list, error_map_dict, word_trie, char_trie, char_set, offset, replace_error_path):
    """
    对错误词进行替换
    @param:
    @return:
    """
    # 打乱数据
    index_list = list(range(len(train_text_list)))
    random.shuffle(index_list)

    # 每个错误词对应的句子数量
    error_word_num_dict = {}
        
    # 检测能够替换的词及位置
    replace_word_obj_list = []
    replace_char_obj_list = []
    num = 0
    for index in index_list:
        text = train_text_list[index] 
        
        # 寻找多字错误
        wrong_word_list = []
        select_word_text = ""
        word_result_set = word_trie.search(text, with_index=True)
        for word, (start_index, end_index) in word_result_set:
            error_word_num_dict[word] = error_word_num_dict.get(word, 0) + 1
            # 某个词的错误数量达到设定阈值后则不再加入
            if error_word_num_dict[word] < 8:
                wrong_word_list.append([word, start_index, end_index])
            
            select_word_text += word
                    
        if len(wrong_word_list) > 0:
            replace_word_obj_list.append({
                "index": index+offset,
                "text": text,
                "errors": wrong_word_list
            })
        
        # 寻找单字错误 未命中多字错误 or 多字错误与单字错误重叠时
        joint_word_set = set(select_word_text) & char_set
        if len(wrong_word_list) == 0 or len(joint_word_set) > 0:
            wrong_char_list = []
            char_result_set = char_trie.search(text, with_index=True)
            for single_word, (start_index, end_index) in char_result_set:
                # 多字错误与单字错误重叠时，只命中重叠字
                if len(wrong_word_list) != 0 and single_word not in joint_word_set:
                    continue
                
                error_word_num_dict[single_word] = error_word_num_dict.get(single_word, 0) + 1
                # 某个字的错误数量达到设定阈值后则不再加入
                if error_word_num_dict[single_word] < 16:
                    wrong_char_list.append([single_word, start_index, end_index])

            if len(wrong_char_list) > 0:
                replace_char_obj_list.append({
                    "index": index+offset,
                    "text": text,
                    "errors": wrong_char_list
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
    FileUtil.write_json_data(replace_obj_list, replace_error_path)

def generate_replace_error(train_data_path, error_map_path, train_replace_error_path):
    """
    检测能生成替换错误的位置
    @param:
    @return:
    """
    train_text_list = FileUtil.read_raw_data(train_data_path)
    error_map_dict = {item.split("\t")[0]: json.loads(item.split("\t")[1]) for item in FileUtil.read_raw_data(error_map_path)}
   
    # 构建词语ac自动机(先去除单字)
    word_trie = ahocorasick.AhoCorasick([word for word in error_map_dict.keys() if len(word) > 1]) 
    # 构建单字ac自动机
    char_trie = ahocorasick.AhoCorasick([word for word in error_map_dict.keys() if len(word) == 1]) 
    
    LogUtil.logger.info("全量数据加载完毕")

    char_set = set([word for word in error_map_dict.keys() if len(word) == 1])
    
    # 单进程处理
    # replace_word_task(train_text_list, error_map_dict, word_trie, char_trie, char_set, 0, train_replace_error_path)

    # 多进程处理
    for i in range(24):
        replace_error_path = train_replace_error_path + "_" + str(i+1)
        offset = 1000000
        p = multiprocessing.Process(target=replace_word_task, args=(train_text_list[i*offset:(i+1)*offset], error_map_dict, word_trie, char_trie, char_set, i*offset, replace_error_path))
        p.start()

if __name__ == "__main__":
    train_data_path = "/ssd1/users/fangzheng/data/mt_error/source_data/pseudo_train/source_train/combine_train.txt"
    error_map_path = "/ssd1/users/fangzheng/data/mt_error/source_data/dict/asr_error_align_mlm.txt"
    train_replace_error_path = "/ssd1/users/fangzheng/data/mt_error/source_data/pseudo_train/replace_error_mlm/train_replace_error"

    generate_replace_error(train_data_path, error_map_path, train_replace_error_path)