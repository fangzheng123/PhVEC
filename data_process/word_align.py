#encoding: utf-8
'''
@File    :   word_align.py
@Time    :   2021/05/25 15:05:09
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''
import sys
sys.path.append("../../MTError")

import json
import math
import re
from collections import Counter

import jieba
from pypinyin import lazy_pinyin 

from util.file_util import FileUtil
from util.log_util import LogUtil

class WordAlign(object):
    """
    构建对齐词典
    """
    def combine_zh_dict(self, large_pinyin_path, ci_json_path, idiom_json_path, combine_dict_path):
        """
        融合中文词典数据
        @param:
        @return:
        """
        word_list = [ele.split(":")[0] for ele in FileUtil.read_raw_data(large_pinyin_path)]
        ci_list = [ele["ci"] for ele in json.load(open(ci_json_path, "r", encoding="utf-8"))]
        idiom_list= [ele["word"] for ele in json.load(open(idiom_json_path, "r", encoding="utf-8"))]

        combine_word_list = list(set(word_list + ci_list + idiom_list))
        word_pinyin_list = [word + "\t" + "".join(lazy_pinyin(word)) for word in combine_word_list]
        FileUtil.write_raw_data(word_pinyin_list, combine_dict_path)
    
    def get_same_pinyin_word(self, combine_dict_path, same_pinyin_path):
        """
        获取相同拼音对应词语
        @param:
        @return:
        """
        word_pinyin_list = FileUtil.read_raw_data(combine_dict_path)
        
        pinyin_word_dict = {}
        for item in word_pinyin_list:
            word, pinyin = item.split("\t")
            pinyin_word_dict.setdefault(pinyin, []).append(word)
        
        pinyin_word_list = [pinyin + "\t" + json.dumps(word_list, ensure_ascii=False) for pinyin, word_list in sorted(pinyin_word_dict.items(), key=lambda x: len(x[1]), reverse=True)]
        FileUtil.write_raw_data(pinyin_word_list, same_pinyin_path)

    def get_word_tfidf(self, thuc_news_path, word_tfidf_path):
        """
        通过大规模新闻语料获取词语TFIDF
        @param:
        @return:
        """
        sent_obj_list = FileUtil.read_json_data(thuc_news_path)
        
        # 切词
        tf_dict = {}
        idf_dict = {}
        for index, sent_obj in enumerate(sent_obj_list): 
            text = sent_obj["text"]
            for word in jieba.cut(text, cut_all=False):
                tf_dict[word] = tf_dict.get(word, 0) + 1
                idf_dict.setdefault(word, set()).add(index)

            if index % 10000 == 0:
                LogUtil.logger.info(index)
        
        word_tf_idf_list = []
        all_sent_num = len(sent_obj_list)
        for word, tf in tf_dict.items():
            idf = math.log(all_sent_num/(len(idf_dict.get(word, set())) + 1))
            word_tf_idf_list.append(word + "\t" + str(tf) + "\t" + str(idf))

            if len(word_tf_idf_list) % 100000 == 0:
                LogUtil.logger.info(len(word_tf_idf_list))

        FileUtil.write_raw_data(word_tf_idf_list, word_tfidf_path)
    
    def get_word_co_occurrence(self, thuc_news_path, word_co_occurrence_path):
        """
        获取每个字周围共现字
        @param:
        @return:
        """
        sent_obj_list = FileUtil.read_json_data(thuc_news_path)
        
        word_co_occurrence_dict = {}
        for sent_index, sent_obj in enumerate(sent_obj_list):
            text = sent_obj["text"]
            text = re.sub("[《》“”()。；?!，：（）、\-/！？=‘’【】 ]", "", text)
            for index, word in enumerate(text[:-1]):
                if word in word_co_occurrence_dict:
                    word_co_occurrence_dict[word][text[index+1]] = word_co_occurrence_dict[word].get(text[index+1], 0) + 1
                else:
                    word_co_occurrence_dict.setdefault(word, {}).setdefault(text[index+1], 1)

            if sent_index % 100000 == 0:
                LogUtil.logger.info(sent_index)
        
        # 对于每个字选取共现top10的字 
        word_occurrence_list = [word + "\t" + json.dumps(dict(Counter(co_occurrence_dict).most_common(10)), ensure_ascii=False) for word, co_occurrence_dict in word_co_occurrence_dict.items()]
        FileUtil.write_raw_data(word_occurrence_list, word_co_occurrence_path)

    def filter_pinyin_word(self, same_pinyin_path, word_tfidf_path, filter_same_pinyin_path):
        """
        过滤拥有相同拼音的词语
        @param:
        @return:
        """
        pinyin_word_dict = {item.split("\t")[0]: json.loads(item.split("\t")[1]) for item in FileUtil.read_raw_data(same_pinyin_path)}
        word_tfidf_dict = {item.split("\t")[0]: {"tf": int(item.split("\t")[1]), \
            "idf": float(item.split("\t")[-1])} for item in FileUtil.read_raw_data(word_tfidf_path) \
                 if len(item.strip().split("\t")) == 3}

        same_word_dict = {}
        for pinyin, word_list in pinyin_word_dict.items():
            filter_word_list = [word for word in word_list if word_tfidf_dict.get(word, {}).get("tf", 0) > 50]
            if len(filter_word_list) > 1:
                for index, word in enumerate(filter_word_list): 
                    same_word_dict.setdefault(word, []).extend(filter_word_list[:index] + filter_word_list[index+1:])
        
        same_word_list = [label + "\t" + json.dumps(confusion_list, ensure_ascii=False) for label, confusion_list in same_word_dict.items()]
        FileUtil.write_raw_data(same_word_list, filter_same_pinyin_path)
    
    def combine_asr_error(self, same_pinyin_path, asr_error_align_path, combine_map_path):
        """
        将相同拼音映射词与asr常见错误映射词合并
        @param:
        @return:
        """
        pinyin_words_dict = {item.split("\t")[0]: json.loads(item.split("\t")[1]) for item in FileUtil.read_raw_data(same_pinyin_path)}
        asr_word_dict = {item.split("\t")[0]: json.loads(item.split("\t")[1]) for item in FileUtil.read_raw_data(asr_error_align_path)}

        # 合并映射
        combine_label_confusion_dict = {label: list(word_dict.keys()) for label, word_dict in asr_word_dict.items()}
        for label, word_list in pinyin_words_dict.items():
            combine_label_confusion_dict.setdefault(label, []).extend(word_list)
            
        combine_list = [label + "\t" + json.dumps(list(set(confusion_list)), ensure_ascii=False) for label, confusion_list in combine_label_confusion_dict.items()]
        FileUtil.write_raw_data(combine_list, combine_map_path)
    
if __name__ == "__main__":
    root_dir = "/ssd1/users/fangzheng/data/mt_error/source_data/dict/zh_cidian/"
    large_pinyin_path = root_dir + "large_pinyin.txt"
    ci_json_path = root_dir + "ci.json"
    idiom_json_path = root_dir + "idiom.json"
    combine_dict_path = root_dir + "combine.txt"
    same_pinyin_path = root_dir + "same_pinyin.txt"
    
    thuc_news_path = "/ssd1/users/fangzheng/data/mt_error/source_data/pseudo_train/thucnews_train.txt"
    word_tfidf_path = root_dir + "word_tfidf.txt"
    word_co_occurrence_path = root_dir + "word_co_occurrence.txt"

    filter_same_pinyin_path = root_dir + "filter_same_pinyin.txt"
    asr_error_path = "/ssd1/users/fangzheng/data/mt_error/source_data/dict/asr_align_filter.txt"
    combine_mapping_path = "/ssd1/users/fangzheng/data/mt_error/source_data/dict/combine_map.txt"

    word_align = WordAlign()
    # word_align.combine_zh_dict(large_pinyin_path, ci_json_path, idiom_json_path, combine_dict_path)
    # word_align.get_same_pinyin_word(combine_dict_path, same_pinyin_path)
    # word_align.get_word_tfidf(thuc_news_path, word_tfidf_path)
    word_align.get_word_co_occurrence(thuc_news_path, word_co_occurrence_path)
    # word_align.filter_pinyin_word(same_pinyin_path, word_tfidf_path, filter_same_pinyin_path)
    # word_align.combine_asr_error(filter_same_pinyin_path, asr_error_path, combine_mapping_path)