#encoding: utf-8
'''
@File    :   config.py
@Time    :   2021/04/19 14:37:26
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

ROOT_DIR = "/ssd1/users/fangzheng/data/mt_error/"

############ 原文本数据  ############
SOURCE_DIR = ROOT_DIR + "source_data/bstc/"

# 训练数据
TRAIN_DIR = SOURCE_DIR + "train/"
TRAIN_DATA_PATH = SOURCE_DIR + "train.txt"
TRAIN_CORRECTION_PATH = SOURCE_DIR + "train_correction.txt"

# 验证数据
DEV_DIR = SOURCE_DIR + "dev/"
DEV_DATA_PATH = SOURCE_DIR + "dev.txt"
DEV_CORRECTION_PATH = SOURCE_DIR + "dev_correction.txt"

# 切分训练数据
TRAIN_SPLIT_PATH = SOURCE_DIR + "train_split_correction.txt"
DEV_SPLIT_PATH = SOURCE_DIR + "dev_split_correction.txt"

############ 格式化数据  ############
FORMAT_DIR = ROOT_DIR + "format_data/"

# 单条句子作为上下文
SINGLE_TRAIN_SENT_PATH = FORMAT_DIR + "train_single_sent.txt"
SINGLE_DEV_SENT_PATH = FORMAT_DIR + "dev_single_sent.txt"
