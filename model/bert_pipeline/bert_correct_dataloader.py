#encoding: utf-8
'''
@File    :   bert_dataloader.py
@Time    :   2021/05/31 16:59:33
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import torch
from datasets import load_dataset
from transformers import BertTokenizer

class BERTCorrectDataLoader(object):
    """
    BERT模型数据加载类
    """
    def __init__(self, args, bert_tokenizer):
        self.args = args
        self.bert_tokenizer = bert_tokenizer
    
    def load_data(self, data_path, is_train=False):
        """
        加载纠错数据
        @param:
        @return:
        """
        def get_correction_token_label(item, correction_token_input):
            """
            获取错误纠正器中token对应的标签
            @param:
            @return:
            """
            # 正确句子
            correction_label = correction_token_input["input_ids"][:]
            if len(item["errors"]) > 0:
                error_item = item["errors"][0]
                label_word = error_item["label_word"]
                correct_input = error_item["correct_input"]
                start, end = error_item["correct_error_range"]
                correct_error_align_indexs = error_item["correct_error_align_index"]

                mask_content = correct_input[:start] + "[MASK]" + correct_input[start:end] + "[MASK]" + correct_input[end:]
                mask_token_list = self.bert_tokenizer.tokenize(mask_content)
                mask_pos_list = [i for i, token in enumerate(mask_token_list) if token == "[MASK]"]

                token_begin = mask_pos_list[0]
                token_end = mask_pos_list[1] - 1                    

                align_offset_list = [int((align_index-start)/2) for align_index in correct_error_align_indexs]

                # 加[CLS], 并将拼音对应label标记为[unused1]
                correction_label[token_begin+1: token_end+1] = [1] * (token_end - token_begin)

                # 给部分汉字及拼音打标
                for align_offset, word in zip(align_offset_list, label_word):
                    if token_begin+1+align_offset < self.args.max_input_len:
                        correction_label[token_begin+1+align_offset] = self.bert_tokenizer.convert_tokens_to_ids(word)

            correction_label = correction_label[:self.args.max_input_len]
            # 最后一位为[SEP]或[PAD]对应的标签, 均为0                 
            correction_label[-1] = 0

            return correction_label

        def tokenize_item_func(item):
            """
            处理单条训练数据
            @param:
            @return:
            """
            # 错误句子输入为加入拼音后的输入
            if len(item["errors"]) == 0:
                # 正确句子输入与detection输入相同
                correction_token_input = self.bert_tokenizer(item["asr"], truncation=True, padding="max_length", max_length=self.args.max_input_len)
            else:
                # 错误句子输入为加入拼音后的输入
                correction_token_input = self.bert_tokenizer(
                    item["errors"][0]["correct_input"], truncation=True, padding="max_length", max_length=self.args.max_input_len)
                
            # 构造标签数据
            correction_label = get_correction_token_label(item, correction_token_input)
            transcript_label = self.bert_tokenizer(item["transcript"], truncation=True, padding="max_length", max_length=self.args.max_input_len)
            correction_token_input["labels"] = correction_label
            correction_token_input["transcript"] = transcript_label["input_ids"]

            return correction_token_input

        # 调用hf的dataset库，将数据转化为tensor
        dataset = load_dataset("json", data_files=data_path, split="train")
        source_column_names = dataset.column_names

        # 构建不同格式数据
        dataset = dataset.map(tokenize_item_func, remove_columns=source_column_names, num_proc=self.args.dataloader_proc_num)
        dataset.set_format(type="torch")

        # 训练数据shuffle
        if is_train:
            dataset = dataset.shuffle(self.args.seed)
            batch_size = self.args.train_batch_size
        else:
            batch_size = self.args.eval_batch_size
        
        # 打印数据结果
        # if is_train:
        #     correct_inputs = self.bert_tokenizer.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        #     correct_labels = self.bert_tokenizer.batch_decode(dataset["labels"], skip_special_tokens=False)
        #     correct_transcripts = self.bert_tokenizer.batch_decode(dataset["transcript"], skip_special_tokens=False)

        #     for correct_ele, correct_label_ele, transcript_ele in zip(correct_inputs, correct_labels, correct_transcripts):
        #         print(correct_ele, "###########", correct_label_ele, "###########", transcript_ele)

        # 将dataset转化为torch的DataLoader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        return dataloader
