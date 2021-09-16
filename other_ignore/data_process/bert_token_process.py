#encoding: utf-8
'''
@File    :   bert_token_process.py
@Time    :   2021/08/23 20:35:09
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

from transformers import BertModel

from util.file_util import FileUtil

class BERTTokenProcess(object):
    """
    BERT token分析
    """
    def output_token(self, data_path):
        """
        输出拼音和汉字token对应的embedding
        @param:
        @return:
        """
        
        data_obj_list = FileUtil.read_json_data(data_path)

        for data_obj in data_obj_list:
            if len(data_obj["errors"]) > 0:
                correct_input = data_obj["correct_input"]
                token_input_dict = self.bert_tokenizer(text, truncation=True, padding="max_length", max_length=48, return_tensors="pt")
                token_input_dict = {k: v.to(self.args.device) for k, v in token_input_dict.items()}


        