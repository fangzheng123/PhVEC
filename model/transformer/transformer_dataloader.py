#encoding: utf-8
'''
@File    :   transformer_dataloader.py
@Time    :   2021/04/27 14:27:50
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import torch
import numpy as np
from datasets import load_dataset

class TransformerDataLoader(object):
    """
    Transformer模型数据加载类
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
        def tokenize_item_func(item):
            """
            处理单条数据
            @param:
            @return:
            """
            asr_input = self.bert_tokenizer(
                item["asr"], truncation=True, padding="max_length", max_length=self.args.max_input_len)
            transcript_label = self.bert_tokenizer(
                    item["transcript"], truncation=True, padding="max_length", max_length=self.args.max_output_len)

            src_padding_mask_list = []
            tgt_padding_mask_list = []
            for src, tgt in zip(asr_input["input_ids"], transcript_label["input_ids"]):
                src_padding_mask = [False if token != self.bert_tokenizer.pad_token_id else True for token in src]
                # tgt输入中不包含最后一个字符
                tgt_padding_mask = [False if token != self.bert_tokenizer.pad_token_id else True for token in tgt[:-1]]
                src_padding_mask_list.append(src_padding_mask)
                tgt_padding_mask_list.append(tgt_padding_mask)

            # # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore padding in the loss.
            # if self.args.ignore_pad_token_for_loss:
            #     asr_input["tgt"] = [[(l if l != self.bert_tokenizer.pad_token_id else -100) for l in label] for label in transcript_label["input_ids"]]

            asr_input["src"] = asr_input["input_ids"]
            asr_input["tgt"] = transcript_label["input_ids"]
            asr_input["src_padding_mask"] = src_padding_mask_list
            asr_input["tgt_padding_mask"] = tgt_padding_mask_list
            
            asr_input.pop("input_ids")
            asr_input.pop("token_type_ids")
            asr_input.pop("attention_mask")
            
            return asr_input
        
        # 调用hf的dataset库，将数据转化为tensor
        dataset = load_dataset("json", data_files=data_path, split="train")
        source_column_names = dataset.column_names
        dataset = dataset.map(tokenize_item_func, batched=True,
                              remove_columns=source_column_names,
                                num_proc=self.args.dataloader_proc_num)
        
        dataset.set_format(type="torch") 
                    
        # 训练数据shuffle
        if is_train:
            dataset = dataset.shuffle(self.args.seed)
            batch_size = self.args.train_batch_size
        else:
            batch_size = self.args.eval_batch_size

        # 将dataset转化为torch的DataLoader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        for batch_data in dataloader:
            for k, v in batch_data.items():
                print(k, type(v), v.shape)
                print(k, v)
            break

        return dataloader
        


    
