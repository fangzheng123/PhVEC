#encoding: utf-8
'''
@File    :   bert_single_dataloader.py
@Time    :   2021/06/07 17:29:15
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import torch
import pyarrow as pa
from datasets import load_dataset
from transformers import BertTokenizer

from util.text_util import TextUtil

class BERTSingleDataLoader(object):
    """
    BERT模型数据加载类
    """
    def __init__(self, args, bert_tokenizer):
        self.args = args
        self.bert_tokenizer = bert_tokenizer
        
    def load_data(self, data_path, is_train=False):
        """
        加载数据
        @param:
        @return:
        """
        def tokenize_train_item_func(item):
            """
            处理单条训练数据
            @param:
            @return:
            """
            if self.args.do_ctc:
                max_target_len = self.args.max_target_len
            else:
                max_target_len = self.args.max_input_len

            # 构造输入数据
            correction_token_input = self.bert_tokenizer(
                TextUtil.filter_symbol(item["asr"]), truncation=True, padding="max_length", max_length=self.args.max_input_len)
            
            # 构造标签数据
            correction_label = self.bert_tokenizer(
                TextUtil.filter_symbol(item["transcript"]), truncation=True, padding="max_length", max_length=max_target_len)

            correction_token_input["labels"] = correction_label["input_ids"]

            # 使用CTC loss时加入input_lengths和target_lengths
            if self.args.do_ctc:
                correction_token_input["input_lengths"] = self.args.max_input_len
                correction_token_input["target_lengths"] = self.args.max_target_len
    
            return correction_token_input

        # 调用hf的dataset库，将数据转化为tensor
        dataset = load_dataset("json", data_files=data_path, split="train")
        source_column_names = dataset.column_names

        dataset = dataset.map(tokenize_train_item_func,remove_columns=source_column_names, num_proc=self.args.dataloader_proc_num)
        dataset.set_format(type="torch")

        # 训练数据shuffle
        if is_train:
            dataset = dataset.shuffle(self.args.seed)
            batch_size = self.args.train_batch_size
        else:
            batch_size = self.args.eval_batch_size
        
        # 打印数据结果
        # if is_train:
        #     correct_inputs = self.bert_tokenizer.batch_decode(dataset["input_ids"], skip_special_tokens=True)
        #     correct_labels = self.bert_tokenizer.batch_decode(dataset["labels"], skip_special_tokens=True)
        #     for correct_ele, correct_label_ele in zip(correct_inputs, correct_labels):
        #         if correct_ele != correct_label_ele:
        #             print(correct_ele, "###########", correct_label_ele)
            
        # 将dataset转化为torch的DataLoader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        return dataloader