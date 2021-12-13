#encoding: utf-8
'''
@File    :   context_process.py
@Time    :   2021/04/21 11:10:39
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import sys
sys.path.append("../../PhVEC")

import re

import config
from util.file_util import FileUtil

class ContextProcess(object):
    """
    上下文处理
    """

    def build_single_context(self, data_path, single_context_path):
        """
        只使用单条句子上下文数据
        @param:
        @return:
        """
        doc_obj_list = FileUtil.read_json_data(data_path)

        all_sent_obj_list = []
        for doc_obj in doc_obj_list:
            sent_list = doc_obj.get("sent_list", [])
            for sent_index, sent_obj in enumerate(sent_list):
                asr_content = "".join([ele[0] for ele in sent_obj.get("asr", []) if len(ele) > 0])
                if len(asr_content) == 0:
                    continue

                all_sent_obj_list.append({
                    "doc_index": doc_obj.get("doc_index", ""),
                    "sent_index": str(sent_index),
                    "transcript": sent_obj.get("transcript", ""),
                    "asr": asr_content
                })
        
        FileUtil.write_json_data(all_sent_obj_list, single_context_path)

if __name__ == "__main__":
    context_process = ContextProcess()
    context_process.build_single_context(config.TRAIN_SPLIT_PATH, config.SINGLE_TRAIN_SENT_PATH)
    context_process.build_single_context(config.DEV_SPLIT_PATH, config.SINGLE_DEV_SENT_PATH)





