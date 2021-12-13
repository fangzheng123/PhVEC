#encoding: utf-8
'''
@File    :   data_analyse.py
@Time    :   2021/04/20 11:10:25
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''
import sys
sys.path.append("../../PhVEC")

import re
import json
from datasets import load_metric

import config
from util.file_util import FileUtil

class DataAnalyse(object):
    """
    数据分析
    """
    def doc_error_static(self, doc_data_path):
        """
        错误asr统计
        @param:
        @return:
        """
        asr_data_list = FileUtil.read_json_data(doc_data_path)
        
        error_num = 0
        all_num = 0
        for doc_obj in asr_data_list:
            for sent_obj in doc_obj.get("sent_list", []):
                transcript = re.sub("。|，|！|？|《|》|“|”", "", sent_obj.get("transcript", ""))   
                asr_result = re.sub("。|，|！|？|《|》|“|”", "", "".join([ele[0] for ele in sent_obj.get("asr", []) if len(ele) > 0]))
                if transcript != asr_result:
                    error_num += 1
                    print(transcript, asr_result)

            print("\n"*2)
            all_num += len(doc_obj.get("sent_list", []))
        
        print(error_num, all_num, error_num/all_num)
    
    def sent_error_static(self, sent_data_path):
        """
        句子级错误分析
        @param:
        @return:
        """
        sent_obj_list = FileUtil.read_json_data(sent_data_path)

        count = 0
        for sent_obj in sent_obj_list:
            if len(sent_obj.get("transcript", "")) > 64:
                print(sent_obj)
                count += 1
        
        print(count, len(sent_obj_list), count/len(sent_obj_list))

    def error_bleu_score(self, sent_data_path):
        """
        分析直接复制情况下的bleu值
        @param:
        @return:
        """
        sent_obj_list = FileUtil.read_json_data(sent_data_path)
        bleu_metric = load_metric("sacrebleu")

        decoded_preds = []
        decoded_labels = []
        for sent_obj in sent_obj_list:
            decoded_preds.append(sent_obj["asr"])
            decoded_labels.append(sent_obj["transcript"])
        
        decoded_preds = [pred.strip().replace(" ", "") for pred in decoded_preds]
        decoded_labels = [[label.strip().replace(" ", "")] for label in decoded_labels]
        bleu_metric.add_batch(predictions=decoded_preds, references=decoded_labels)
        eval_bleu_score = bleu_metric.compute()["score"]
        
        print(eval_bleu_score)
        return eval_bleu_score
    
    def static_confusion_num(self, confusion_path):
        """
        统计混淆集数量
        """
        confusion_data_list = FileUtil.read_raw_data(confusion_path)
        total_num = sum([len(json.loads(item.strip().split("\t")[-1])) for item in confusion_data_list])
        print(total_num)

    def get_sent_length(self, data_path):
        """
        获取句子平均长度
        @param:
        @return:
        """
        data_obj_list = FileUtil.read_json_data(data_path)
        sent_len_list = [len(data_obj["asr"]) for data_obj in data_obj_list]

        print(sum(sent_len_list)/len(sent_len_list))


       
if __name__ == "__main__":
    data_analyse = DataAnalyse()
    # data_analyse.sent_error_static(config.SINGLE_SENT_PATH)
    # data_analyse.error_bleu_score(config.SINGLE_DEV_SENT_PATH)
    # data_analyse.static_confusion_num("/ssd1/users/fangzheng/data/mt_error/source_data/dict/asr_align_filter.txt")
    data_analyse.get_sent_length("/ssd1/users/fangzheng/data/mt_error/asr_format/bstc_format/train_dev_format.txt")