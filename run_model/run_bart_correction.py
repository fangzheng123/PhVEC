#encoding: utf-8
'''
@File    :   run_bart_correction.py
@Time    :   2021/04/20 16:29:18
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import sys
sys.path.append("../MTError")

from datasets import load_dataset
from transformers import (
    set_seed,
    HfArgumentParser,
    MBartTokenizer,
    MBartForConditionalGeneration
)

from util.arg_util import BARTArguments
from util.log_util import LogUtil
from model.mbart.bart_dataloader import BARTDataLoader
from model.mbart.bart import BARTModel
from model.mbart.bart_process import BARTProcess

class BARTCorrection(object):
    """
    基于mBART模型的纠错
    """
    def __init__(self, args):
        self.args = args

        self.bart_tokenizer = MBartTokenizer.from_pretrained(self.args.pretrain_model_path, src_lang="zh_CN", tgt_lang="zh_CN")
        self.bart_dataloader = BARTDataLoader(self.args, self.bart_tokenizer)
        # self.bart_model = MBartForConditionalGeneration.from_pretrained(args.pretrain_model_path)
        self.bart_model = BARTModel(self.args)
        self.bart_process = BARTProcess(self.args, self.bart_model, self.bart_tokenizer)

    def train(self):
        """
        训练模型
        @param:
        @return:
        """
        # 加载数据，返回DataLoader
        LogUtil.logger.info("Loading data...")
        train_dataloader = self.bart_dataloader.load_data(self.args.train_data_path, is_train=True)
        dev_dataloader = self.bart_dataloader.load_data(self.args.dev_data_path, is_train=False)
        LogUtil.logger.info("Finished loading data ...")

        # 在初始化模型前固定种子，保证每次运行结果一致
        set_seed(self.args.seed)

        # 训练模型
        LogUtil.logger.info("Training model...")
        self.bart_process.train(train_dataloader, dev_dataloader)
        LogUtil.logger.info("Finished Training model!!!")

    def test(self):
        """
        测试模型
        @param:
        @return:
        """
        # 加载数据
        LogUtil.logger.info("Loading data...")
        test_dataloader = self.bart_dataloader.load_data(self.args.test_data_path, is_train=False)
        LogUtil.logger.info("Finished loading data!!!")

        # 固定种子，保证每次运行结果一致
        set_seed(self.args.seed)

        # 测试模型
        LogUtil.logger.info("Testing model...")
        self.bart_process.test(test_dataloader)


    def predict(self):
        """
        预测
        @param:
        @return:
        """
        pass

if __name__ == "__main__":
    # 初始化huggingface ArgumentParser, 将包含自定义和公用的参数
    args = HfArgumentParser(BARTArguments).parse_args_into_dataclasses()[0]
    args.num_train_epochs = int(args.num_train_epochs)
    
    bart_correction = BARTCorrection(args)
    
    # 模型训练
    if args.do_train:
        bart_correction.train()

    # 模型测试，有真实标签
    if args.do_eval:
        bart_correction.test()

    # 模型预测，无真实标签
    if args.do_predict:
        bart_correction.predict()
