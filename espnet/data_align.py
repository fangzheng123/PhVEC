#encoding: utf-8
'''
@File    :   data_align.py
@Time    :   2021/05/18 13:58:32
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''
import sys
sys.path.append("../../MTError")

import os
import json

from util.asr_score_util import ASRScoreUtil
from util.file_util import FileUtil
from util.log_util import LogUtil

class DataAlign(object):
    """
    将ASR结果与transcript数据对齐
    """
    def align_asr_transcript(self, asr_dir, transcript_data_path, format_data_path, data_type):
        """
        对齐asr结果与transcript结果, 并计算CER
        @param:
        @return:
        """
        # 读取transcript数据
        if data_type == "aishell":
            transcript_data_list = FileUtil.read_raw_data(transcript_data_path)
            transcript_dict = {ele.split()[0]: "".join(ele.split()[1:]) for ele in transcript_data_list}
        elif data_type == "magic":
            transcript_data_list = FileUtil.read_raw_data(transcript_data_path)
            transcript_dict = {ele.split("\t")[0].split(".")[0]: ele.split("\t")[-1] for ele in transcript_data_list[1:]}
        elif data_type == "primeword":
            transcript_data_list = json.load(open(transcript_data_path, "r", encoding="utf-8"))
            transcript_dict = {ele["file"].split(".")[0]: ele["text"].replace(" ", "") for ele in transcript_data_list}
        
        sent_obj_list = []
        for doc_root, _, doc_names in os.walk(asr_dir):
            for doc_name in doc_names:
                doc_path = os.path.join(doc_root, doc_name)
                doc_json = json.load(open(doc_path, "r", encoding="utf-8"))

                for sent_id, asr_sent in doc_json.items():
                    if asr_sent == "":
                        continue
                    sent_obj_list.append({
                        "doc_id": doc_name.split(".")[0],
                        "sent_id": sent_id,
                        "asr": asr_sent,
                        "transcript":  transcript_dict.get(sent_id, "")
                    })

        sent_obj_list = sorted(sent_obj_list, key=lambda x: x["sent_id"])

        # 将格式化后的数据存入文件
        FileUtil.write_json_data(sent_obj_list, format_data_path)

        # 计算CER
        cer = ASRScoreUtil.calculate_cer([ele["asr"] for ele in sent_obj_list], [ele["transcript"] for ele in sent_obj_list])
        LogUtil.logger.info("CER is {0:.2f}%".format(cer*100))

    
if __name__ == "__main__":
    asr_dir = "/ssd1/users/fangzheng/data/asr_data/magic_data/asr/test"
    transcript_data_path = "/ssd1/users/fangzheng/data/asr_data/magic_data/transcript/test_trans.txt"
    format_data_path = "/ssd1/users/fangzheng/data/mt_error/asr_format/magic_format/test_single_format.txt"
    
    data_align = DataAlign()
    data_align.align_asr_transcript(asr_dir, transcript_data_path, format_data_path, "magic")
    

