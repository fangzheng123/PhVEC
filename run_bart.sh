#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=5

export HF_DATASETS_CACHE="/data/fangzheng/project_data/phvec/cache"

# 根路径
ROOT_DIR="/data/fangzheng/project_data/phvec"

# 预训练模型类型
MODEL_TYPE="Bart-base-chinese"

# 预训练模型路径
PRE_TRAINED_MODEL_DIR=$ROOT_DIR/pretrain_model/${MODEL_TYPE}/
# 微调模型存储路径
FINETUNE_MODEL_DIR=$ROOT_DIR/model/bart_model/
FINETUNE_MODEL_PATH=$ROOT_DIR/model/bart_model/bart_cn.pk

LOG_DIR=log
# 创建相关目录
echo ${green}=== Mkdir ===${reset}
mkdir -p $FINETUNE_MODEL_DIR
mkdir -p $LOG_DIR

####################用户需提供的数据#####################
# 模型训练、验证、测试文件
FORMAT_DATA_DIR=$ROOT_DIR/correct_format
TRAIN_DATA_PATH=$FORMAT_DATA_DIR/pseudo_joint_train_format.txt
DEV_DATA_PATH=$FORMAT_DATA_DIR/aishell_dev.txt
TEST_DATA_PATH=$FORMAT_DATA_DIR/aishell_test_100.txt

# 日志
LOG_FILE=$LOG_DIR/bart_test_log.txt

nohup python -u run_model/run_bart_correction.py \
  --do_train \
  --pretrain_model_path=$PRE_TRAINED_MODEL_DIR \
  --output_dir=$FINE_TUNING_MODEL_DIR \
  --model_save_path=$FINETUNE_MODEL_PATH \
  --logging_dir=$LOG_DIR \
  --train_data_path=$TRAIN_DATA_PATH \
  --dev_data_path=$DEV_DATA_PATH \
  --test_data_path=$TEST_DATA_PATH \
  --dataloader_proc_num=8 \
  --num_train_epochs=5 \
  --per_device_train_batch_size=300 \
  --per_device_eval_batch_size=1 \
  --eval_batch_step=2000 \
  --require_improvement=10000 \
  --max_input_len=48 \
  --max_output_len=52 \
  --learning_rate=5e-5 \
  --weight_decay=0.01 \
  --warmup_ratio=0.1 \
  --num_beams=1 \
  --seed=42 \
  > $LOG_FILE 2>&1 &