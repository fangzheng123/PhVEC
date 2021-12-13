#encoding: utf-8
'''
@File    :   asr_inference.py
@Time    :   2021/05/17 15:42:04
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import sys
sys.path.append("../../PhVEC")

import os
import json
import argparse

import torch
import soundfile
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text

from util.log_util import LogUtil

class ASRProcess(object):
    """
    ASR模型推断
    """
    def __init__(self, pre_train_model_path, cache_dir):
        """
        此处使用GPU进行解码，CPU解码速度太慢，因此在运行sh脚本时记得设定GPU卡号
        @param:
        @return:
        """
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
        try:
            speech, rate = soundfile.read(wav_path)
            nbests = self.speech2text(speech)
            text, *_ = nbests[0]
        except:
            LogUtil.logger.error("inference error file: {0}".format(wav_path))
        finally:
            text = ""
        
        return text

    def inference_doc(self, doc_obj, asr_result_dir):
        """
        预测一篇文档内所有句子对应的语音文件，一个文档中所有句子语音文件均在一个文件夹中
        @param:
        @return:
        """
        sent_dict = {}
        doc_name, sent_path_list = doc_obj
        doc_path = asr_result_dir + doc_name + ".json"

        # 解码过的重复文件不再解码
        if os.path.exists(doc_path):
            LogUtil.logger.info("{0} --- exists ----", doc_path)
            return

        for sent_path in sent_path_list:
            text = asr_process.inference_wav_file(sent_path)
            sent_dict[sent_path.split("/")[-1].split(".")[0]] = text
                    
        json.dump(sent_dict, open(doc_path, "w", encoding="utf-8"), ensure_ascii=False) 
                        
    def inference_dir(self, wav_data_dir, asr_result_dir):
        """
        解码一个文件夹下所有的文档语音文件
        @param:
        @return:
        """
        doc_list = []
        for doc_root, doc_names, _ in os.walk(wav_data_dir):
            for doc_name in doc_names:
                for sent_root, _, sent_wav_names in os.walk(os.path.join(doc_root, doc_name)):
                    sent_list = [os.path.join(sent_root, sent_wav) for sent_wav in sent_wav_names]
                    doc_list.append((doc_name, sent_list))
        LogUtil.logger.info("路径遍历完毕")
        
        for doc_obj in doc_list:
            self.inference_doc(doc_obj, asr_result_dir)

if __name__ == "__main__":
    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre_train_model_path", default=None, type=str, required=True)
    parser.add_argument("--cache_dir", default=None, type=str, required=True)
    parser.add_argument("--wav_data_dir", default=None, type=str, required=True)
    parser.add_argument("--asr_result_dir", default=None, type=str, required=True)
    args = parser.parse_args()

    asr_process = ASRProcess(args.pre_train_model_path, args.cache_dir)
    asr_process.inference_dir(args.wav_data_dir, args.asr_result_dir)
    
