#encoding: utf-8
'''
@File    :   pseudo_process.py
@Time    :   2021/05/24 11:14:52
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''
import sys
sys.path.append("../../MTError")

import os
import re
from xml.dom import minidom

import Levenshtein
from pypinyin import lazy_pinyin

from util.file_util import FileUtil

class TrainDataPrepare(object):
    """
    准备训练数据
    """

    def extract_aishell_train(self, transcript_data_path, dev_test_path, train_data_path):
        """
        从aishell中抽取训练数据
        @param:
        @return:
        """
        # 读取transcript数据
        transcript_data_list = FileUtil.read_raw_data(transcript_data_path)
        transcript_dict = {ele.split()[0]: "".join(ele.split()[1:]) for ele in transcript_data_list}
        
        train_obj_list = []
        dev_test_obj_list = FileUtil.read_json_data(dev_test_path)
        dev_test_set = {ele["sent_id"] for ele in dev_test_obj_list}
        for sent_id, sent in transcript_dict.items():
            if sent_id not in dev_test_set:
                train_obj_list.append({
                    "doc_id": sent_id[6:11],
                    "sent_id": sent_id,
                    "transcript":  sent
                    })
        FileUtil.write_json_data(train_obj_list, train_data_path)
    
    def extract_tencent_sgml(self, sgml_path, cec_train_path):
        """
        解析sgml数据
        @param:
        @return:
        """
        dom_tree = minidom.parse(sgml_path)
        # 文档根元素
        root_node = dom_tree.documentElement
        sent_objs = root_node.getElementsByTagName("SENTENCE")
        
        data_obj_list = []
        for sent_obj in sent_objs:
            sent = sent_obj.getElementsByTagName("TEXT")[0].childNodes[0].data
            
            mistake_obj_list = []
            for mistake_obj in sent_obj.getElementsByTagName("MISTAKE"):
                location = mistake_obj.getElementsByTagName("LOCATION")[0].childNodes[0].data
                wrong = mistake_obj.getElementsByTagName("WRONG")[0].childNodes[0].data
                correction = mistake_obj.getElementsByTagName("CORRECTION")[0].childNodes[0].data
                mistake_obj_list.append({
                    "loc": location,
                    "wrong": wrong,
                    "correction": correction
                })
            
            data_obj_list.append({
                "sent": sent,
                "mistake": mistake_obj_list
            })
        
        FileUtil.write_json_data(data_obj_list, cec_train_path)
    
    def filter_tencent_data(self, source_path, filter_path):
        """
        过滤非拼音错误数据
        @param:
        @return:
        """
        data_obj_list = FileUtil.read_json_data(source_path)
        
        filter_data_list = []
        for data_obj in data_obj_list:
            sent = data_obj["sent"]

            filter_mistake_list = []
            for mistake_obj in data_obj["mistake"]:
                wrong = mistake_obj["wrong"]
                correction = mistake_obj["correction"]
                wrong_pinyin, correction_pinyin = "".join(lazy_pinyin(wrong)), "".join(lazy_pinyin(correction))
                # 非拼音错误，则保持原状
                if Levenshtein.ratio(wrong_pinyin, correction_pinyin) < 0.8:
                    loc = int(mistake_obj["loc"]) - 1
                    sent = sent[:loc] + correction + sent[loc+1:]
                else:
                    mistake_obj["loc"] = int(mistake_obj["loc"]) - 1
                    filter_mistake_list.append(mistake_obj)
            
            if len(filter_mistake_list) > 0:
                filter_data_list.append({
                    "sent": sent,
                    "mistake": filter_mistake_list
                })
        
        print(len(data_obj_list), len(filter_data_list))
        FileUtil.write_json_data(filter_data_list, filter_path)
    
    def extract_thuc_news(self, data_dir, train_path):
        """
        处理清华新闻数据
        @param:
        @return:
        """
        text_obj_list = []
        for type_root, type_names, _ in os.walk(data_dir):
            for type_name in type_names:
                if type_name not in {"体育", "娱乐", "时政", "教育", "社会", "科技", "财经"}:
                    continue
                for doc_root, _, doc_names in os.walk(os.path.join(type_root, type_name)):
                    for doc_name in doc_names:
                        doc_path = os.path.join(doc_root, doc_name)
                        text_list = [text.replace(" ", "") for text in FileUtil.read_raw_data(doc_path) if text.replace(" ", "") != ""]
                        
                        for index, text in enumerate(text_list): 
                            doc_id = doc_name.split(".")[0]
                            text_obj_list.append({
                                "doc_id": doc_id,
                                "sent_id": doc_id + "_" + str(index),
                                "text": text,
                                "type": type_name,
                            })
                
        FileUtil.write_json_data(text_obj_list, train_path)
    
    def split_thuc_news(self, thuc_path, split_thuc_path):
        """
        对清华数据进行切分
        @param:
        @return:
        """
        text_obj_list = FileUtil.read_json_data(thuc_path)
        
        pattern = re.compile(r'(?<=\().+?(?=\))') 
        new_text_obj_list = []
        for text_obj in text_obj_list:
            text = text_obj["text"]
            split_list = re.split("。|；|？|!|，|：", text)
            
            # 将长度过短的句子与之后的句子进行拼接
            index = 0
            last_text = ""
            new_split_list = []
            while index < len(split_list):
                cur_text = last_text + split_list[index]
                if len(cur_text) > 9 or last_text != "" or index == len(split_list) - 1:
                    new_split_list.append(cur_text)
                    last_text = ""                    
                else:
                    last_text = split_list[index]

                index += 1
            
            for index, split_text in enumerate(new_split_list):
                rep_text = pattern.sub("", split_text)
                rep_text = re.sub("[《》“”()]", "", rep_text)
                if len(rep_text) < 6 or len(rep_text) > 30:
                    continue

                new_text_obj_list.append({
                    "doc_id": text_obj["doc_id"],
                    "sent_id": text_obj["sent_id"] + "_" + str(index),
                    "text": rep_text,
                    "type": text_obj["type"]
                })
            
        FileUtil.write_json_data(new_text_obj_list, split_thuc_path)
    
    def extract_bstc_train(self, bstc_align_path, bstc_train_path):
        """
        抽取bstc训练数据
        @param:
        @return:
        """
        doc_obj_list = FileUtil.read_json_data(bstc_align_path)
        
        data_obj_list = []
        for doc_obj in doc_obj_list:
            doc_id = doc_obj["doc_index"]
            for index, sent_obj in enumerate(doc_obj["sent_list"]): 
                data_obj_list.append({
                    "doc_id": doc_id, 
                    "sent_id": doc_id + "_" + str(index),
                    "text": sent_obj["transcript"]
                })
        
        FileUtil.write_json_data(data_obj_list, bstc_train_path)

    def combine_thuc_aishell_bstc(self, thuc_path, aishell_path, bstc_path, combine_path):
        """
        合并清华，aishell，bstc数据
        @param:
        @return:
        """
        thuc_text_list = [ele["text"] for ele in FileUtil.read_json_data(thuc_path)]
        aishell_text_list = [ele["transcript"] for ele in FileUtil.read_json_data(aishell_path)]
        bstc_text_list = [ele["text"] for ele in FileUtil.read_json_data(bstc_path)]

        combine_text_list = thuc_text_list + aishell_text_list + bstc_text_list
        FileUtil.write_raw_data(combine_text_list, combine_path)


if __name__ == "__main__":
    # AISHELL训练数据
    aishell_trancript_path = "/ssd1/users/fangzheng/data/asr_data/data_aishell/transcript/aishell_transcript_v0.8.txt"
    aishell_asr_data_path = "/ssd1/users/fangzheng/data/mt_error/asr_format/aishell_format/dev_test_format.txt"
    aishell_train_path = "/ssd1/users/fangzheng/data/mt_error/source_data/pseudo_train/aishell_train.txt"

    # EMNLP2018构造的数据
    tencent_sgml_path = "/ssd1/users/fangzheng/data/mt_error/source_data/tencent/wang_train.sgml"
    tencent_train_path = "/ssd1/users/fangzheng/data/mt_error/source_data/tencent/wang_train.txt"
    tencent_filter_path = "/ssd1/users/fangzheng/data/mt_error/source_data/pseudo_train/wang_train_filter.txt"
    
    # 清华新闻数据
    thucnews_data_dir = "/ssd1/users/fangzheng/data/THUCNews"
    thucnews_train_path = "/ssd1/users/fangzheng/data/mt_error/source_data/pseudo_train/thucnews_train.txt"
    thucnews_train_split_path = "/ssd1/users/fangzheng/data/mt_error/source_data/pseudo_train/thucnews_train_split.txt"
    
    # BSTC数据
    bstc_align_path = "/ssd1/users/fangzheng/data/mt_error/source_data/bstc/train_correction.txt"
    bstc_train_path = "/ssd1/users/fangzheng/data/mt_error/source_data/pseudo_train/bstc_train.txt"

    # 清华, AISHELL, BSTC融合数据
    combine_train_path = "/ssd1/users/fangzheng/data/mt_error/source_data/pseudo_train/combine_train.txt"

    train_prepare = TrainDataPrepare()
    # train_prepare.extract_sgml(aishell_trancript_path, aishell_asr_data_path, aishell_train_path, "aishell")
    # train_prepare.extract_tencent_sgml(tencent_sgml_path, tencent_train_path)
    # train_prepare.filter_tencent_data(tencent_train_path, tencent_filter_path)
    # train_prepare.extract_thuc_news(thucnews_data_dir, thucnews_train_path)
    # train_prepare.split_thuc_news(thucnews_train_path, thucnews_train_split_path)
    # train_prepare.extract_bstc_train(bstc_align_path, bstc_train_path)

    train_prepare.combine_thuc_aishell_bstc(thucnews_train_split_path, aishell_train_path, bstc_train_path, combine_train_path)
    


        




