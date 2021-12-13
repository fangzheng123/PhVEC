#encoding: utf-8
'''
@File    :   run_bert_joint_pinyin.py
@Time    :   2021/08/24 19:33:35
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import sys
sys.path.append("../PhVEC")

import torch
from datasets import load_dataset
from transformers import (
    set_seed,
    HfArgumentParser,
    BertTokenizer
)

from model.bert_joint_pinyin.bert_pinyin_dataloader import BERTPinyinDataLoader
from model.bert_joint_pinyin.bert_pinyin_model import BERTPinyinModel
from model.bert_joint_pinyin.bert_pinyin_process import BERTPinyinProcess
from util.arg_util import BERTArguments
from util.log_util import LogUtil
from util.file_util import FileUtil

class BERTPinyinCorrection(object):
    """
    基于BERT模型的纠错, 新增Pinyin Token
    用于对比实验
    """
    def __init__(self, args):
        self.args = args
        
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.args.pretrain_model_path)
        # 增加额外的拼音token
        self.bert_tokenizer.add_tokens(FileUtil.read_raw_data(args.pinyin_token_path))

        self.bert_dataloader = BERTPinyinDataLoader(self.args, self.bert_tokenizer)
        
        self.bert_model = BERTPinyinModel(self.args, self.bert_tokenizer)
        self.bert_process = BERTPinyinProcess(self.args, self.bert_model, self.bert_tokenizer)

    def train(self):
        """
        训练模型
        @param:
        @return:
        """
        # 加载数据，返回DataLoader
        LogUtil.logger.info("Loading data...")
        train_dataloader = self.bert_dataloader.load_data(self.args.train_data_path, is_train=True)
        dev_dataloader = self.bert_dataloader.load_data(self.args.dev_data_path, is_train=False)
        LogUtil.logger.info("Finished loading data ...")

        # 在初始化模型前固定种子，保证每次运行结果一致
        set_seed(self.args.seed)

        # 训练模型
        LogUtil.logger.info("Training model...")
        self.bert_process.train(train_dataloader, dev_dataloader)
        LogUtil.logger.info("Finished Training model!!!")

    def test(self):
        """
        测试模型
        @param:
        @return:
        """
        # 加载数据
        LogUtil.logger.info("Loading data...")
        test_dataloader = self.bert_dataloader.load_data(self.args.test_data_path, is_train=False)
        LogUtil.logger.info("Finished loading data!!!")

        # 固定种子，保证每次运行结果一致
        set_seed(self.args.seed)

        # 测试模型
        LogUtil.logger.info("Testing model...")
        self.bert_process.test(test_dataloader)

    def predict(self):
        """
        预测
        @param:
        @return:
        """
        pass

if __name__ == "__main__":
    # 初始化huggingface ArgumentParser, 将包含自定义和公用的参数
    args = HfArgumentParser(BERTArguments).parse_args_into_dataclasses()[0]
    args.num_train_epochs = int(args.num_train_epochs)
    
    bert_pinyin_correction = BERTPinyinCorrection(args)
    
    # 模型训练
    if args.do_train:
        bert_pinyin_correction.train()

    # 模型测试，有真实标签
    if args.do_eval:
        bert_pinyin_correction.test()

    # 模型预测，无真实标签
    if args.do_predict:
        bert_pinyin_correction.predict() 