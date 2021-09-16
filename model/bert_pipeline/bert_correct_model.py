#encoding: utf-8
'''
@File    :   bert.py
@Time    :   2021/05/31 17:01:22
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import torch
import torch.nn as nn
from transformers import BertModel

class BERTCorrectModel(nn.Module):
    """
    使用BERT模型纠错
    """
    
    def __init__(self, args):
        super().__init__()

        self.bert = BertModel.from_pretrained(args.pretrain_model_path)
        
        self.correct_label_num = args.correct_label_num
        self.dropout = nn.Dropout(args.dropout)
        self.correction_classifier = nn.Linear(args.bert_hidden_size, args.correct_label_num)

    def forward(self, x):
        input_ids, attention_mask, token_type_ids = x
        sequence_output, _ = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, return_dict=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.correction_classifier(sequence_output)
        return logits

        




    
    