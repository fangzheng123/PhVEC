#encoding: utf-8
'''
@File    :   fairseq_process.py
@Time    :   2021/05/11 15:30:37
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import sys
sys.path.append("../../MTError")

from transformers import (
    BertTokenizer
)

from util.file_util import FileUtil
from util.log_util import LogUtil
from util.asr_score_util import ASRScoreUtil
from util.text_util import TextUtil

class FairseqProcess(object):
    """
    处理fairseq数据
    """
    def format_data(self, source_data_path, pretrain_model_path, asr_data_path, transcript_data_path, is_bstc=False):
        """
        格式化为fairseq数据
        @param:
        @return:
        """
        all_data_list = FileUtil.read_json_data(source_data_path)
        bert_tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

        all_asr_list = []
        all_transcipt_list = []
        for i, data_obj in enumerate(all_data_list):
            asr = data_obj["asr"]
            transcript = data_obj["transcript"]

            if is_bstc:
                asr = TextUtil.filter_symbol(asr)
                transcript = TextUtil.filter_symbol(transcript)

            # 使用BertTokenizer切词，这样就可以不用BPE来构建词表并切词了，直接使用BERT的词表并切词
            asr_token_list = bert_tokenizer.tokenize(asr)
            transcript_token_list = bert_tokenizer.tokenize(transcript)

            all_asr_list.append(" ".join(asr_token_list))
            all_transcipt_list.append(" ".join(transcript_token_list))

            if i % 10000 == 0:
                LogUtil.logger.info(i)
        
        FileUtil.write_raw_data(all_asr_list, asr_data_path)
        FileUtil.write_raw_data(all_transcipt_list, transcript_data_path)

    def get_cer_result(self, decode_data_path, is_bstc=False, is_variable=False):
        """
        获取transformer模型cer分数
        @param:
        @return:
        """
        raw_data_list = FileUtil.read_raw_data(decode_data_path)

        pred_data_dict = {}
        asr_data_dict = {}
        transcirpt_data_dict = {}
        for raw_data in raw_data_list:
            raw_data = raw_data.strip()
            if raw_data[:2] == "S-":
                if len(raw_data.split("\t")) == 1:
                    data_index = raw_data
                    text = ""
                else:
                    data_index, text = raw_data.split("\t")
                asr_data_dict[int(data_index[2:])] = text.replace(" ", "")
            elif raw_data[:2] == "T-":
                if len(raw_data.split("\t")) == 1:
                    data_index = raw_data
                    text = ""
                else:
                    data_index, text = raw_data.split("\t")
                transcirpt_data_dict[int(data_index[2:])] = text.replace(" ", "")
            elif raw_data[:2] == "D-":
                if len(raw_data.split("\t")) == 2:
                    data_index, score = raw_data.split("\t")
                    text = ""
                else:
                    data_index, score, text = raw_data.split("\t")
                pred_data_dict[int(data_index[2:])] = text.replace(" ", "")
                
        pred_data_list = [text for _, text in sorted(pred_data_dict.items(), key=lambda x: x[0])]
        asr_data_list = [text for _, text in sorted(asr_data_dict.items(), key=lambda x: x[0])]
        transcript_data_list = [text for _, text in sorted(transcirpt_data_dict.items(), key=lambda x: x[0])]

        if is_bstc:
            new_pred_list = []
            new_transcript_data = []
            for pred_data, transcript_data in zip(pred_data_list, transcript_data_list):
                if len(transcript_data) > 37:
                    continue
                new_pred_list.append(pred_data)
                new_transcript_data.append(transcript_data)
            pred_data_list = new_pred_list
            transcript_data_list = new_transcript_data

        # 14329  6756
        len_list = []
        if is_variable:
            new_pred_list = []
            new_transcript_data = []
            new_asr_data_list = []
            text_set = {ele["asr"] for ele in FileUtil.read_json_data("/ssd1/users/fangzheng/data/mt_error/correct_format/magic_test_filter_variable.txt")}
            for asr_data, transcript_data, pred_data in zip(asr_data_list[6756:], transcript_data_list[6756:], pred_data_list[6756:]):
                if asr_data in text_set:
                    new_pred_list.append(pred_data)
                    new_transcript_data.append(transcript_data)
                    len_list.append(abs(len(pred_data) - len(transcript_data)))
                    new_asr_data_list.append(asr_data)
            pred_data_list = new_pred_list
            transcript_data_list = new_transcript_data
            asr_data_list = new_asr_data_list
            # print(len(pred_data_list))
            # print(len([ele for ele in len_list if ele == 0]), len([ele for ele in len_list if ele == 0])/len(len_list))

        for asr, pred, trans in zip(asr_data_list, pred_data_list, new_transcript_data):
            print(asr)
            print(pred)
            print(trans)
            print("")
                
        cer = ASRScoreUtil.calculate_wer(pred_data_list[:], transcript_data_list[:])
        print(cer)

if __name__ == "__main__":
    root_dir = "/ssd1/users/fangzheng/data/mt_error/"
    pretrain_dir = "/ssd1/users/fangzheng/data/mt_error/pretrain_model/BERT-wwm-ext"

    source_path = root_dir + "correct_format/bstc_dev_format.txt"
    # 使用fairseq中翻译模型，相当于源语言为zh, 目标语言为en
    asr_path = root_dir + "fairseq_bstc_format/test.zh"
    transcript_path = root_dir + "fairseq_bstc_format/test.en"

    fair_process = FairseqProcess()
    fair_process.format_data(source_path, pretrain_dir, asr_path, transcript_path, is_bstc=False)
    # fair_process.get_cer_result("/ssd1/users/fangzheng/project/MTError/log/fairseq_levt_transformer_test_log2.txt", is_variable=True)
        

        


        





