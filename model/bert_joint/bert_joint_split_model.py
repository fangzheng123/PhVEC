#encoding: utf-8
'''
@File    :   bert_joint_split_model.py
@Time    :   2021/06/15 21:07:01
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import torch
import torch.nn as nn
from transformers import BertModel

class BERTJointSplitModel(nn.Module):
    """
    使用BERT模型纠错, 不共享编码器
    """
    def __init__(self, args):
        super().__init__()

        self.detect_bert = BertModel.from_pretrained(args.pretrain_model_path)
        self.correct_bert = BertModel.from_pretrained(args.pretrain_model_path)

        self.detect_label_num = args.detect_label_num
        self.correct_label_num = args.correct_label_num
        self.dropout = nn.Dropout(args.dropout)
        self.detection_classifier = nn.Linear(args.bert_hidden_size, args.detect_label_num)
        self.correction_classifier = nn.Linear(args.bert_hidden_size, args.correct_label_num)

    def forward(self, x, is_detect=False):
        input_ids, attention_mask, token_type_ids = x
        if is_detect:
            sequence_output, _ = self.detect_bert(input_ids=input_ids, token_type_ids=token_type_ids, \
                attention_mask=attention_mask, return_dict=False)
            sequence_output = self.dropout(sequence_output)
            logits = self.detection_classifier(sequence_output)
        else:
            sequence_output, _ = self.correct_bert(input_ids=input_ids, token_type_ids=token_type_ids, \
                attention_mask=attention_mask, return_dict=False)
            sequence_output = self.dropout(sequence_output)
            logits = self.correction_classifier(sequence_output)
        
        return logits