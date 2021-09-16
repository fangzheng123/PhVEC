#encoding: utf-8
'''
@File    :   dataloader.py
@Time    :   2021/04/20 16:25:08
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import torch
from datasets import load_dataset
from transformers import MBartTokenizer

class BARTDataLoader(object):
    """
    mBART模型数据加载类
    """
    def __init__(self, args, bart_tokenizer):
        self.args = args
        self.bart_tokenizer = bart_tokenizer

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
            asr_input = self.bart_tokenizer(
                item["asr"], truncation=True, padding="max_length", max_length=self.args.max_input_len)
            with self.bart_tokenizer.as_target_tokenizer():
                transcript_label = self.bart_tokenizer(
                    item["transcript"], truncation=True, padding="max_length", max_length=self.args.max_output_len)

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore padding in the loss.
            if self.args.ignore_pad_token_for_loss:
                transcript_label["input_ids"] = [[(l if l != self.bart_tokenizer.pad_token_id else -100) for l in label] for label in transcript_label["input_ids"]]

            asr_input["labels"] = transcript_label["input_ids"]
            return asr_input

        # 调用hf的dataset库，将数据转化为tensor
        dataset = load_dataset("json", data_files=data_path, split="train")
        source_column_names = dataset.column_names
        dataset = dataset.map(tokenize_item_func, batched=True,
                              remove_columns=source_column_names, num_proc=self.args.dataloader_proc_num)
        dataset.set_format(type="torch")

        # 训练数据shuffle
        if is_train:
            dataset = dataset.shuffle(self.args.seed)
            batch_size = self.args.train_batch_size
        else:
            batch_size = self.args.eval_batch_size
                
        # decoded_inputs = self.bart_tokenizer.batch_decode(dataset["input_ids"], skip_special_tokens=True)
        # decoded_labels = self.bart_tokenizer.batch_decode(dataset["labels"], skip_special_tokens=True)
        # for input_ele, label_ele in zip(decoded_inputs, decoded_labels):
        #     print(input_ele.encode("utf-8").decode("latin1"), "###########", label_ele.encode("utf-8").decode("latin1"))

        # 将dataset转化为torch的DataLoader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        return dataloader
