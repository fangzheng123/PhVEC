#encoding: utf-8
'''
@File    :   bert_detect_dataloader.py
@Time    :   2021/06/09 16:35:26
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import torch
from datasets import load_dataset
from transformers import BertTokenizer

class BERTDetectDataLoader(object):
    """
    BERT模型数据加载类
    """
    def __init__(self, args, bert_tokenizer):
        self.args = args
        self.bert_tokenizer = bert_tokenizer
        
    def load_data(self, data_path, is_train=False, is_predict=False):
        """
        加载检测数据
        @param:
        @return:
        """
        def get_detection_token_label(item):
            """
            获取错误检测器中token对应的标签
            @param:
            @return:
            """
            asr_text = item["asr"]

            # 正确句子
            detection_label = [1] * self.args.max_input_len
            if len(item["errors"]) > 0:
                start, end = item["errors"][0]["detect_error_range"]
                mask_content = asr_text[:start] + "[MASK]" + asr_text[start:end] + "[MASK]" + asr_text[end:]
                mask_token_list = self.bert_tokenizer.tokenize(mask_content)

                mask_pos_list = [i for i, token in enumerate(mask_token_list) if token == "[MASK]"]

                token_begin = mask_pos_list[0]
                token_end = mask_pos_list[1] - 1

                # 加[CLS]
                detection_label[token_begin+1: token_end+1] = [0] * (token_end-token_begin)

            detection_label = detection_label[:self.args.max_input_len]
            # 最后一位为[SEP]或[PAD]对应的标签, 均为1                   
            detection_label[-1] = 1

            return detection_label

        def tokenize_item_func(item):
            """
            处理单条训练数据
            @param:
            @return:
            """
            # 构造输入数据
            detection_token_input = self.bert_tokenizer(
                item["asr"], truncation=True, padding="max_length", max_length=self.args.max_input_len)
            
            transcript_label = self.bert_tokenizer(item["transcript"], truncation=True, padding="max_length", max_length=self.args.max_input_len)

            # 构造标签数据
            if not is_predict:
                detection_label = get_detection_token_label(item)
                detection_token_input["labels"] = detection_label
            
            detection_token_input["transcript"] = transcript_label["input_ids"]

            return detection_token_input

        # 调用hf的dataset库，将数据转化为tensor
        dataset = load_dataset("json", data_files=data_path, split="train")
        source_column_names = dataset.column_names
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
        #     detect_inputs = self.bert_tokenizer.batch_decode(dataset["input_ids"], skip_special_tokens=True)
        #     detect_labels = dataset["labels"]
        #     detect_transcripts = self.bert_tokenizer.batch_decode(dataset["transcript"], skip_special_tokens=True)

        #     for detect_ele, detect_label_ele, transcript_ele in zip(detect_inputs, detect_labels, detect_transcripts):
        #         print(detect_ele, "###########", detect_label_ele, "###########", transcript_ele)
            
        # 将dataset转化为torch的DataLoader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        return dataloader
