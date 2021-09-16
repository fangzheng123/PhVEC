#encoding: utf-8
'''
@File    :   bstc_dev_process.py
@Time    :   2021/08/10 16:58:49
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import sys
sys.path.append("../../MTError")

import os

from util.file_util import FileUtil

class BSTCDevProcess(object):
    """
    BSTC DEV数据处理
    """
    def read_asr_data(self, asr_path):
        """
        读取asr文件数据
        @param:
        @return:
        """
        raw_list = FileUtil.read_raw_data(asr_path)

        doc_sent_dict = {}
        for item in raw_list:
            doc_name, sent = item.split(":")
            if "-" not in doc_name:
                continue
            doc_sent_dict.setdefault(doc_name.split("-")[0], {})\
                .setdefault(doc_name.split("-")[1].replace(".wav", ""), sent)

        return doc_sent_dict

    def read_transcript_data(self, transcript_path):
        """
        读取单个transcript文件
        @param:
        @return:
        """
        data_obj_list = FileUtil.read_json_data(transcript_path)

        doc_name = ""
        sent_dict = {}
        for i, data_obj in enumerate(data_obj_list):
            doc_name = data_obj["wav_id"]
            sent_dict[str(i)] = data_obj["transcript"]

        return doc_name, sent_dict

    def combine_asr_transcript(self, asr_path, transcript_dir, asr_format_path):
        """
        整合asr和transcript数据
        @param:
        @return:
        """
        asr_doc_sent_dict = self.read_asr_data(asr_path)

        doc_trans_dict = {}
        for root, dirs, files in os.walk(transcript_dir):
            for file_name in files:
                doc_name, sent_dict = self.read_transcript_data(os.path.join(root, file_name))
                doc_trans_dict[doc_name] = sent_dict
        
        sent_obj_list = []
        for doc_name, trans_sent_dict in doc_trans_dict.items():
            for sent_index, trans in trans_sent_dict.items():
                sent_obj_list.append({
                    "doc_index": doc_name,
                    "sent_index": sent_index,
                    "transcript": trans,
                    "asr": asr_doc_sent_dict[doc_name][sent_index]
                })
        FileUtil.write_json_data(sent_obj_list, asr_format_path)

if __name__ == "__main__":
    asr_path = "/ssd1/users/fangzheng/data/asr_data/bstc/dev_asr.txt"
    transcript_dir = "/ssd1/users/fangzheng/data/asr_data/bstc/dev"
    asr_format_path = "/ssd1/users/fangzheng/data/mt_error/asr_format/bstc_format/dev_single_format.txt"

    bstc_process = BSTCDevProcess()
    bstc_process.combine_asr_transcript(asr_path, transcript_dir, asr_format_path)

