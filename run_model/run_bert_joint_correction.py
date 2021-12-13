#encoding: utf-8
'''
@File    :   run_bert_correction.py
@Time    :   2021/05/31 16:54:18
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

from util.arg_util import BERTArguments
from util.log_util import LogUtil
from util.model_util import ModelUtil
from model.bert_joint.bert_joint_dataloader import BERTJointDataLoader
from model.bert_joint.bert_joint_model import BERTJointModel
from model.bert_joint.bert_joint_process import BERTJointProcess
from model.bert_joint.bert_joint_split_model import BERTJointSplitModel
from model.bert_joint.bert_joint_split_process import BERTJointSplitProcess
from model.bert_joint.bert_joint_ctc_process import BERTJointCTCProcess

class BERTJointCorrection(object):
    """
    基于BERT模型的纠错, 包括错误探测及错误预测
    """
    def __init__(self, args):
        self.args = args

        self.bert_tokenizer = BertTokenizer.from_pretrained(self.args.pretrain_model_path)
        self.bert_dataloader = BERTJointDataLoader(self.args, self.bert_tokenizer)
        
        # 不共享检测和纠错的BERT编码器
        if self.args.do_split:
            self.bert_model = BERTJointSplitModel(self.args)
            self.bert_process = BERTJointSplitProcess(self.args, self.bert_model, self.bert_tokenizer)
        else:
            self.bert_model = BERTJointModel(self.args)

            if self.args.do_ctc:
                self.bert_process = BERTJointCTCProcess(self.args, self.bert_model, self.bert_tokenizer)
            else:
                self.bert_process = BERTJointProcess(self.args, self.bert_model, self.bert_tokenizer)

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
    # import os
    # cpu_num = 4
    # os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
    # os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    # os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
    # os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    # os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    # torch.set_num_threads(cpu_num)

    # 初始化huggingface ArgumentParser, 将包含自定义和公用的参数
    args = HfArgumentParser(BERTArguments).parse_args_into_dataclasses()[0]
    args.num_train_epochs = int(args.num_train_epochs)
    
    bert_joint_correction = BERTJointCorrection(args)
    
    # 模型训练
    if args.do_train:
        bert_joint_correction.train()

    # 模型测试，有真实标签
    if args.do_eval:
        bert_joint_correction.test()

    # 模型预测，无真实标签
    if args.do_predict:
        bert_joint_correction.predict()
