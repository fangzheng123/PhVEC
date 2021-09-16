#encoding: utf-8
'''
@File    :   bert_detect_model.py
@Time    :   2021/06/09 16:34:48
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import torch
import torch.nn as nn
from transformers import BertModel

class BERTDetectModel(nn.Module):
    """
    使用BERT模型探测错误位置
    """
    def __init__(self, args):
        super().__init__()

        self.bert = BertModel.from_pretrained(args.pretrain_model_path)
        
        self.detect_label_num = args.detect_label_num
        self.dropout = nn.Dropout(args.dropout)
        self.detection_classifier = nn.Linear(args.bert_hidden_size, args.detect_label_num)

    def forward(self, x):
        input_ids, attention_mask, token_type_ids = x
        sequence_output, _ = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, return_dict=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.detection_classifier(sequence_output)
        return logits