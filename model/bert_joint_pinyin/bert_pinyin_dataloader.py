#encoding: utf-8
'''
@File    :   bert_pinyin_dataloader.py
@Time    :   2021/08/24 19:38:33
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import torch
import pyarrow as pa
from datasets import load_dataset
from transformers import BertTokenizer

from util.text_util import TextUtil

class BERTPinyinDataLoader(object):
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

            # 截取长度
            detection_label = detection_label[:self.args.max_input_len]
            # 最后一位为[SEP]或[PAD]对应的标签, 均为1                   
            detection_label[-1] = 1

            return detection_label
        
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
            # 当[SEP]被截取掉时，将最后一位转化为[SEP]                    
            if correction_label[-1] not in [0, 102]:
                correction_label[-1] = 102

            return correction_label

        def tokenize_train_item_func(item):
            """
            处理单条训练数据
            @param:
            @return:
            """
            # 构造输入数据
            detection_token_input = self.bert_tokenizer(
                item["asr"], truncation=True, padding="max_length", max_length=self.args.max_input_len)
            
            if len(item["errors"]) == 0:
                # 正确句子输入与detection输入相同
                correction_token_input = detection_token_input
                correction_label = detection_token_input
            else:
                # 错误句子输入为加入拼音后的输入
                correction_token_input = self.bert_tokenizer(
                    item["errors"][0]["correct_input"], truncation=True, padding="max_length", max_length=self.args.max_input_len)
                correction_label = self.bert_tokenizer(
                    item["errors"][0]["correct_label"], truncation=True, padding="max_length", max_length=self.args.max_input_len)

            # 构造标签数据
            detection_label = get_detection_token_label(item)
            
            if self.args.do_ctc:
                max_target_len = self.args.max_target_len
            else:
                max_target_len = self.args.max_input_len
            transcript_label = self.bert_tokenizer(item["transcript"], truncation=True, padding="max_length", max_length=max_target_len)

            own_data_dict = {
                "detect_input_ids": detection_token_input["input_ids"], 
                "detect_attention_mask": detection_token_input["attention_mask"], 
                "detect_type_ids": detection_token_input["token_type_ids"], 
                "correct_input_ids": correction_token_input["input_ids"], 
                "correct_attention_mask": correction_token_input["attention_mask"], 
                "correct_type_ids": correction_token_input["token_type_ids"], 
                "detect_labels": detection_label, 
                "correct_labels": correction_label["input_ids"],
                "transcript_labels": transcript_label["input_ids"], 
            }

            # 此处不能忽略padding loss, 若忽视则[PAD]对应的位置会乱生成词
            # if self.args.ignore_pad_token_for_loss:
            #     own_data_dict["detect_labels"] = [l if mask == 1 else -100 for l, mask in zip(own_data_dict["detect_labels"], own_data_dict["detect_attention_mask"])]
            #     own_data_dict["correct_labels"] = [l if l != self.bert_tokenizer.pad_token_id else -100 for l in own_data_dict["correct_labels"]]

            return own_data_dict
        
        def tokenize_test_item_func(item):
            """
            处理单条测试数据
            @param:
            @return:
            """
            # 构造输入数据
            detection_token_input = self.bert_tokenizer(item["asr"], truncation=True, padding="max_length", max_length=self.args.max_input_len)
            
            # 构造标签数据
            if self.args.do_ctc:
                max_target_len = self.args.max_target_len
            else:
                max_target_len = self.args.max_input_len
            transcript_label = self.bert_tokenizer(item["transcript"], truncation=True, padding="max_length", max_length=max_target_len)
            
            own_data_dict = {
                "detect_input_ids": detection_token_input["input_ids"], 
                "detect_attention_mask": detection_token_input["attention_mask"], 
                "detect_type_ids": detection_token_input["token_type_ids"], 
                "transcript_labels": transcript_label["input_ids"], 
            }
            
            return own_data_dict

        # 调用hf的dataset库，将数据转化为tensor
        dataset = load_dataset("json", data_files=data_path, split="train")
        source_column_names = dataset.column_names

        # 构建不同格式数据
        if is_train:
            dataset = dataset.map(tokenize_train_item_func,remove_columns=source_column_names, num_proc=self.args.dataloader_proc_num)
        else:
            dataset = dataset.map(tokenize_test_item_func,remove_columns=source_column_names, num_proc=self.args.dataloader_proc_num)

        dataset.set_format(type="torch")

        # 训练数据shuffle
        if is_train:
            dataset = dataset.shuffle(self.args.seed)
            batch_size = self.args.train_batch_size
        else:
            batch_size = self.args.eval_batch_size
        
        # 打印数据结果
        # if is_train:
        #     detect_inputs = self.bert_tokenizer.batch_decode(dataset["detect_input_ids"], skip_special_tokens=True)
        #     correct_inputs = self.bert_tokenizer.batch_decode(dataset["correct_input_ids"], skip_special_tokens=True)
        #     detect_labels = dataset["detect_labels"]
        #     correct_labels = self.bert_tokenizer.batch_decode(dataset["correct_labels"], skip_special_tokens=False)
        #     for detect_ele, correct_ele, detect_label_ele, correct_label_ele in zip(detect_inputs, correct_inputs, detect_labels, correct_labels):
        #         if detect_ele != correct_ele:
        #             print(detect_ele, "###########", correct_ele, "###########", detect_label_ele, "###########", correct_label_ele)

        # 将dataset转化为torch的DataLoader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        return dataloader