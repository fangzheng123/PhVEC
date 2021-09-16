#encoding: utf-8
'''
@File    :   BART.py
@Time    :   2021/04/20 14:23:26
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import torch
import torch.nn as nn
from transformers import MBartForConditionalGeneration, MBartTokenizer, CONFIG_MAPPING

class BARTModel(nn.Module):
    """
    使用BART模型纠错
    """
    
    def __init__(self, args):
        super().__init__()
        self.bart_pretrain_model = MBartForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    
    def forward(self, **kwags):
        seq2seq_output = self.bart_pretrain_model(**kwags, output_hidden_states=True)
        return seq2seq_output

    def generate(self, **kwags):
        """
        生成结果id
        @param:
        @return:
        """
        return self.bart_pretrain_model.generate(**kwags)

