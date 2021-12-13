# encoding: utf-8
'''
@File    :   pre_process.py
@Time    :   2021/04/19 14:56:23
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''
import sys
sys.path.append("../../PhVEC")

import os
import json
import random

import config
from util.file_util import FileUtil

class PreProcess(object):
    """
    数据预处理
    """
    def __init__(self):
        pass

    def combine_train_data(self, data_dir, combine_path):
        """
        融合训练集合中多个文件的数据
        @param:
        @return:
        """
        # 读取transcript数据
        transcrip_dir = data_dir + "transcription_translation"
        transcript_data_dict = {}
        for root, dirs, files in os.walk(transcrip_dir):
            for file_name in files:
                transcript_data_dict[file_name.split(".")[0]] = FileUtil.read_json_data(
                    os.path.join(root, file_name))

        # 读取ASR数据
        asr_dir = data_dir + "asr/asr_sentences"
        asr_data_dict = {}
        for doc_root, doc_names, _ in os.walk(asr_dir):
            for doc_name in doc_names:
                for sent_root, _, sents in os.walk(os.path.join(doc_root, doc_name)):
                    for sent_name in sents:
                        asr_sent_list = FileUtil.read_json_data(
                            os.path.join(sent_root, sent_name))
                        asr_data_dict.setdefault(doc_name, {}).setdefault(
                            sent_name, asr_sent_list)
        # 对ASR文档中句子按照序号排序
        for doc_name, sent_dict in asr_data_dict.items():
            asr_data_dict[doc_name] = [sent_list for name, sent_list in sorted(
                sent_dict.items(), key=lambda x: int(x[0].split(".")[0].split("-")[-1]))]

        # 融合transcript和asr数据
        combine_doc_list = []
        for doc_name, doc_golden_list in transcript_data_dict.items():
            asr_list = asr_data_dict.get(doc_name)
            doc_dict = {
                "doc_index": doc_name,
                "sent_num": len(doc_golden_list),
                "sent_list": [{"transcript": golden, "asr": asr} for golden, asr in zip(doc_golden_list, asr_list)]
            }
            combine_doc_list.append(doc_dict)

        FileUtil.write_json_data(combine_doc_list, combine_path)

    def get_train_correction_data(self, source_data_path, filter_data_path):
        """
        获取纠错训练数据
        @param:
        @return:
        """
        source_data_list = FileUtil.read_json_data(source_data_path)

        filter_data_list = []
        for data_json in source_data_list:
            new_sent_list = []
            for item in data_json.get("sent_list", []):
                transcript_obj = item.get("transcript", {})

                # 解析asr结果，可能是字符串也可能是列表
                asr_list = []
                for ele in item.get("asr", []):
                    if type(ele) is list:
                        if len(ele) > 0:
                            ele = ele[0]
                        else:
                            continue
                    asr_result = ele.get("results_recognition", [])
                    if type(asr_result) is list:
                        asr_list.append(asr_result)
                    elif type(asr_result) is str:
                        asr_list.append([asr_result])

                new_sent_list.append({
                    "transcript": transcript_obj.get("transcript", ""),
                    "asr": asr_list,
                    "translation": transcript_obj.get("translation", "")
                })

            filter_data_list.append({
                "doc_index": data_json.get("doc_index", ""),
                "sent_num": data_json.get("sent_num", ""),
                "sent_list": new_sent_list
            })

        FileUtil.write_json_data(
            filter_data_list, filter_data_path, is_indent=False)

    def read_dev_transcript_data(self, dev_transcript_path):
        """
        读取dev transcript数据
        @param:
        @return:
        """
        sent_list = []

        last_sent = ""
        with open(dev_transcript_path, "r", encoding="utf-8") as dev_transcript_file:
            for item in dev_transcript_file:
                item = item.strip()
                if len(item) < len(last_sent) and last_sent not in item:
                    sent_list.append(last_sent)
                last_sent = item
            sent_list.append(last_sent)

        return sent_list

    def read_dev_asr_data(self, dev_asr_path):
        """
        读取dev asr数据
        @param:
        @return:
        """
        sent_list = []

        with open(dev_asr_path, "r", encoding="utf-8") as dev_asr_file:
            for item in dev_asr_file:
                item = item.strip()
                ele_list = item.split(",")
                if "final" in ele_list[-3]:
                    sent_list.append(ele_list[1].split(":")[-1].strip())

        return sent_list

    def combine_dev_data(self, data_dir, combine_path):
        """
        融合dev数据
        """
        # 读取transcript数据
        transcrip_dir = data_dir + "streaming_transcription"
        transcript_data_dict = {}
        for root, dirs, files in os.walk(transcrip_dir):
            for file_name in files:
                transcript_data_dict[file_name.split(".")[0]] = self.read_dev_transcript_data(
                    os.path.join(root, file_name))

        # 读取asr数据
        asr_dir = data_dir + "streaming_asr"
        asr_data_dict = {}
        for root, dirs, files in os.walk(asr_dir):
            for file_name in files:
                asr_data_dict[file_name.split(".")[0]] = self.read_dev_asr_data(
                    os.path.join(root, file_name))

        # 融合transcript和asr数据
        combine_doc_list = []
        for doc_name, doc_golden_list in transcript_data_dict.items():
            asr_list = asr_data_dict.get(doc_name, [])
            doc_dict = {
                "doc_index": doc_name,
                "transcript_sent_num": len(doc_golden_list),
                "asr_sent_num": len(asr_list),
                "transcript_list": doc_golden_list,
                "asr_list": asr_list
            }
            combine_doc_list.append(doc_dict)

        FileUtil.write_json_data(combine_doc_list, combine_path, is_indent=True)

    def get_dev_correction_data(self, dev_data_path, dev_correction_data_path):
        """
        获取纠错dev数据
        @param:
        @return:
        """
        doc_obj_list = FileUtil.read_json_data(dev_data_path)

        for doc_obj in doc_obj_list:
            # 对齐transcript和asr句子
            transcript_sent_list = doc_obj.get("transcript_list", [])
            asr_sent_list = doc_obj.get("asr_list", [])

            trans_index = 0
            asr_index = 0
            last_asr_sent = ""
            align_asr_list = []
            while asr_index < len(asr_sent_list) and trans_index < len(transcript_sent_list):
                trans_sent = transcript_sent_list[trans_index]
                # 上一条asr被切分，则继续对齐
                if last_asr_sent == "":
                    asr_sent = asr_sent_list[asr_index]
                else:
                    asr_sent = last_asr_sent

                # 句子内容或长度相近，则直接对齐
                if (abs(len(trans_sent) - len(asr_sent)) < 3 and len(
                    set(trans_sent) & set(asr_sent)) > len(trans_sent)/2) or (
                    abs(len(trans_sent) - len(asr_sent)) < 6 and (
                        len(set(trans_sent[-3:]) & set(asr_sent[-3:])) > 1 or len(
                            set(trans_sent[:3]) & set(asr_sent[:3])) > 1)):
                    align_asr_list.append(asr_sent)
                    trans_index += 1
                    asr_index += 1
                    last_asr_sent = ""
                # asr长于transcript, 对asr进行切分
                elif len(asr_sent) - len(trans_sent) > 3:
                    curr_asr_sent = asr_sent[:len(trans_sent)]
                    last_asr_sent = asr_sent[len(trans_sent):]

                    # 使用下一句的头部切分
                    flag = True
                    if trans_index+1 < len(transcript_sent_list):
                        next_trans_sent = transcript_sent_list[trans_index+1]
                        head_word = next_trans_sent[:2]
                        if head_word in asr_sent:
                            split_index = asr_sent[::-1].index(head_word[::-1])
                            if len(asr_sent) - split_index - len(trans_sent) > 3:
                                continue
                            curr_asr_sent = asr_sent[:-split_index-1]
                            last_asr_sent = asr_sent[-split_index-1:] 
                            flag = False

                    # 使用当前句的尾部切分
                    if flag:
                        tail_word = trans_sent[-2:]
                        if tail_word in asr_sent:
                            split_index = asr_sent.index(tail_word)
                            if split_index - len(trans_sent) > 3:
                                continue
                            curr_asr_sent = asr_sent[:split_index+2]
                            last_asr_sent = asr_sent[split_index+2:]
                    
                    align_asr_list.append(curr_asr_sent)
                    trans_index += 1
                # transcript长于asr, asr与下一条拼接后继续对齐
                elif len(trans_sent) - len(asr_sent) > 3:
                    if asr_index+1 < len(asr_sent_list):
                        last_asr_sent = asr_sent + asr_sent_list[asr_index+1]
                    asr_index += 1
                # 兜底对齐
                else:
                    align_asr_list.append(asr_sent)
                    trans_index += 1
                    asr_index += 1
                    last_asr_sent = ""
                    # print(asr_sent+ "__asr", trans_sent)

            min_len = min(len(align_asr_list), len(transcript_sent_list))
            for align, golden in zip(align_asr_list[:min_len], transcript_sent_list[:min_len]):
                print("####".join([align, golden]))
            print(len(align_asr_list), len(transcript_sent_list), len(asr_sent_list))
            print("\n" * 3)
            # break
        
    def split_train_dev_data(self, data_path, train_split_path, dev_split_path):
        """
        将数据切分为train和dev数据
        @param:
        @return:
        """
        all_data_list = FileUtil.read_json_data(data_path)
        
        dev_index_set = set(random.choices(range(len(all_data_list)), k=15))
        dev_data_list = [item for index, item in enumerate(all_data_list) if index in dev_index_set]
        train_data_list = [item for index, item in enumerate(all_data_list) if index not in dev_index_set]

        FileUtil.write_json_data(train_data_list, train_split_path)
        FileUtil.write_json_data(dev_data_list, dev_split_path)

if __name__ == "__main__":
    pre_process = PreProcess()
    pre_process.combine_train_data(config.TRAIN_DIR, config.TRAIN_DATA_PATH)
    # pre_process.get_train_correction_data(config.TRAIN_DATA_PATH, config.TRAIN_CORRECTION_PATH)

    pre_process.combine_dev_data(config.DEV_DIR, config.DEV_DATA_PATH)
    # pre_process.get_dev_correction_data(config.DEV_DATA_PATH, config.DEV_CORRECTION_PATH)

    # pre_process.split_train_dev_data(config.TRAIN_CORRECTION_PATH, config.TRAIN_SPLIT_PATH, config.DEV_SPLIT_PATH)
