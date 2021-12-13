#encoding: utf-8
'''
@File    :   run_transformer_correction.py
@Time    :   2021/04/27 17:41:01
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import sys
sys.path.append("../PhVEC")

from datasets import load_dataset
from transformers import (
    set_seed,
    HfArgumentParser,
    BertTokenizer,
)

from util.arg_util import TransformerArguments
from util.log_util import LogUtil
from model.transformer.transformer_dataloader import TransformerDataLoader
from model.transformer.transformer_model import Seq2SeqTransformer
from model.transformer.transformer_process import TransformerProcess

class TransformerCorrection(object):
    """
    基于Transformer模型的纠错
    """
    def __init__(self, args):
        self.args = args

        self.bert_tokenizer = BertTokenizer.from_pretrained(self.args.pretrain_model_path)
        self.transformer_dataloader = TransformerDataLoader(self.args, self.bert_tokenizer)
        self.transformer_model = Seq2SeqTransformer(self.args).to(self.args.device)
        self.transformer_process = TransformerProcess(self.args, self.transformer_model, self.bert_tokenizer)

    def train(self):
        """
        训练模型
        @param:
        @return:
        """
        # 加载数据，返回DataLoader
        LogUtil.logger.info("Loading data...")
        train_dataloader = self.transformer_dataloader.load_data(self.args.train_data_path, is_train=True)
        dev_dataloader = self.transformer_dataloader.load_data(self.args.dev_data_path, is_train=False)
        LogUtil.logger.info("Finished loading data ...")

        # 在初始化模型前固定种子，保证每次运行结果一致
        set_seed(self.args.seed)

        # 训练模型
        LogUtil.logger.info("Training model...")
        self.transformer_process.train(train_dataloader, dev_dataloader)
        LogUtil.logger.info("Finished Training model!!!")

    def test(self):
        """
        测试模型
        @param:
        @return:
        """
        # 加载数据
        LogUtil.logger.info("Loading data...")
        test_dataloader = self.transformer_dataloader.load_data(self.args.test_data_path, is_train=False)
        LogUtil.logger.info("Finished loading data!!!")

        # 固定种子，保证每次运行结果一致
        set_seed(self.args.seed)

        # 测试模型
        LogUtil.logger.info("Testing model...")
        self.transformer_process.test(test_dataloader)


    def predict(self):
        """
        预测
        @param:
        @return:
        """
        pass

if __name__ == "__main__":
    # 初始化huggingface ArgumentParser, 将包含自定义和公用的参数
    args = HfArgumentParser(TransformerArguments).parse_args_into_dataclasses()[0]
    args.num_train_epochs = int(args.num_train_epochs)
    
    transformer_correction = TransformerCorrection(args)
    
    # 模型训练
    if args.do_train:
        transformer_correction.train()

    # 模型测试，有真实标签
    if args.do_eval:
        transformer_correction.test()

    # 模型预测，无真实标签
    if args.do_predict:
        transformer_correction.predict()
