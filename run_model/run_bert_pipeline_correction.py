#encoding: utf-8
'''
@File    :   run_bert_joint_correction2.py
@Time    :   2021/06/09 14:36:05
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import sys
sys.path.append("../PhVEC")

from datasets import load_dataset
from transformers import (
    set_seed,
    HfArgumentParser,
    BertTokenizer
)

from util.arg_util import BERTPipelineArguments
from util.log_util import LogUtil
from model.bert_pipeline.bert_detect_dataloader import BERTDetectDataLoader
from model.bert_pipeline.bert_correct_dataloader import BERTCorrectDataLoader
from model.bert_pipeline.bert_detect_model import BERTDetectModel
from model.bert_pipeline.bert_correct_model import BERTCorrectModel
from model.bert_pipeline.bert_detect_process import BERTDetectProcess
from model.bert_pipeline.bert_correct_process import BERTCorrectProcess

class BERTPipelineCorrection(object):
    """
    基于BERT模型的纠错, 包括错误探测及错误预测
    """
    def __init__(self, args):
        self.args = args

        self.bert_tokenizer = BertTokenizer.from_pretrained(self.args.pretrain_model_path)

        if self.args.do_detect:
            self.bert_dataloader = BERTDetectDataLoader(self.args, self.bert_tokenizer)
            self.bert_detect_model = BERTDetectModel(self.args)
            self.bert_process = BERTDetectProcess(self.args, self.bert_detect_model, self.bert_tokenizer)

        if self.args.do_correct:
            self.bert_dataloader = BERTCorrectDataLoader(self.args, self.bert_tokenizer)
            self.bert_correct_model = BERTCorrectModel(self.args)
            self.bert_process = BERTCorrectProcess(self.args, self.bert_correct_model, self.bert_tokenizer)

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
        # 加载数据
        LogUtil.logger.info("Loading data...")
        bert_detect_dataloader = BERTDetectDataLoader(self.args, self.bert_tokenizer)
        detect_test_dataloader = bert_detect_dataloader.load_data(self.args.test_data_path, is_train=False, is_predict=True)
        LogUtil.logger.info("Finished loading detect data!!!")

        # 预测错误的句子
        bert_detect_model = BERTDetectModel(self.args)
        bert_detect_process = BERTDetectProcess(self.args, bert_detect_model, self.bert_tokenizer)
        right_input_list, all_correct_input_tuple = bert_detect_process.predict(detect_test_dataloader)

        # 对需要纠正的句子进行纠错
        bert_correct_model = BERTCorrectModel(self.args)
        bert_correct_process = BERTCorrectProcess(self.args, bert_correct_model, self.bert_tokenizer)
        cer_score = bert_correct_process.predict(right_input_list, all_correct_input_tuple)

        LogUtil.logger.info("CER Score:{0}".format(cer_score))



if __name__ == "__main__":
    # 初始化huggingface ArgumentParser, 将包含自定义和公用的参数
    args = HfArgumentParser(BERTPipelineArguments).parse_args_into_dataclasses()[0]
    args.num_train_epochs = int(args.num_train_epochs)
    
    bert_pipeline_correction = BERTPipelineCorrection(args)
    
    # 模型训练
    if args.do_train:
        bert_pipeline_correction.train()

    # 模型测试，有真实标签
    if args.do_eval:
        bert_pipeline_correction.test()

    # 模型预测，无真实标签
    if args.do_predict:
        bert_pipeline_correction.predict()