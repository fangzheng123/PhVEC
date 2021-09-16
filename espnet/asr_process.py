#encoding: utf-8
'''
@File    :   asr_inference.py
@Time    :   2021/05/17 15:42:04
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import sys
sys.path.append("../../MTError")

import os
import json
import argparse
import numpy as np
# from multiprocessing import Pool

import torch
import soundfile
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text
# from util.log_util import LogUtil

class ASRProcess(object):
    """
    ASR模型推断
    """
    def __init__(self, pre_train_model_path, cache_dir):
        # 加载预训练模型
        self.speech2text = Speech2Text(
            **ModelDownloader(cache_dir).download_and_unpack(pre_train_model_path),
            maxlenratio=0.0,
            minlenratio=0.0,
            beam_size=10,
            ctc_weight=0.6,
            lm_weight=0.3,
            penalty=0.0,
            nbest=1,
            device="cuda"
        )

    def inference_wav_file(self, wav_path):
        """
        预测单个语音文件
        @param:
        @return:
        """
        text = ""
        
        try:
            speech, rate = soundfile.read(wav_path)
            nbests = self.speech2text(speech)
            text, *_ = nbests[0]
        except:
            print("error", wav_path)
        
        return text

    def inference_doc(self, doc_obj, asr_result_dir):
        """
        预测doc内的语音文件
        @param:
        @return:
        """
        sent_dict = {}
        doc_name, sent_path_list = doc_obj
        doc_path = asr_result_dir + doc_name + ".json"
        if os.path.exists(doc_path):
            print(doc_path + "--- exists ----")
            return

        for sent_path in sent_path_list:
            text = asr_process.inference_wav_file(sent_path)
            sent_dict[sent_path.split("/")[-1].split(".")[0]] = text
            print(text)
                    
        json.dump(sent_dict, open(doc_path, "w", encoding="utf-8"), ensure_ascii=False) 
                        
    def inference_dir(self, wav_data_dir, asr_result_dir):
        doc_list = []
        for doc_root, doc_names, _ in os.walk(wav_data_dir):
            for doc_name in doc_names:
                for sent_root, _, sent_wav_names in os.walk(os.path.join(doc_root, doc_name)):
                    sent_list = [os.path.join(sent_root, sent_wav) for sent_wav in sent_wav_names]
                    doc_list.append((doc_name, sent_list))

        print("路径遍历完毕")
        for doc_obj in doc_list:
            self.inference_doc(doc_obj, asr_result_dir)

if __name__ == "__main__":
    pre_train_model_path = "/ssd1/users/fangzheng/data/asr_data/pretrian_model/asr_train_asr_conformer3_raw_char_batch_bins4000000_accum_grad4_sp_valid.acc.ave.zip"
    cache_dir = "/ssd1/users/fangzheng/data/asr_data/pretrian_model/espnet"
    
    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_data_dir", default=None, type=str, required=True)
    parser.add_argument("--asr_result_dir", default=None, type=str, required=True)
    args = parser.parse_args()
    
    # wav_data_dir = "/ssd1/users/fangzheng/data/asr_data/data_aishell/wav/train_3"
    # asr_result_dir = "/ssd1/users/fangzheng/data/asr_data/data_aishell/asr/train/"

    asr_process = ASRProcess(pre_train_model_path, cache_dir)
    asr_process.inference_dir(args.wav_data_dir, args.asr_result_dir)
    
