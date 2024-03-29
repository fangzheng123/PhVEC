export CUDA_VISIBLE_DEVICES=3

# 根路径
ROOT_DIR="/ssd1/users/fangzheng/data/mt_error"

# 预训练模型类型
MODEL_TYPE="BERT-wwm-ext"

# 预训练模型路径
PRE_TRAINED_MODEL_DIR=$ROOT_DIR/pretrain_model/${MODEL_TYPE}/
# 微调模型存储路径
FINETUNE_MODEL_DIR=$ROOT_DIR/model/bert_joint_model
FINETUNE_MODEL_PATH=$FINETUNE_MODEL_DIR/bert_compare_complete_token.pk

LOG_DIR=log
# 创建相关目录
echo ${green}=== Mkdir ===${reset}
mkdir -p $FINETUNE_MODEL_DIR
mkdir -p $LOG_DIR

####################用户需提供的数据#####################
# 模型训练、验证、测试文件
FORMAT_DATA_DIR=$ROOT_DIR/compare_format
TRAIN_DATA_PATH=$FORMAT_DATA_DIR/pseudo_joint_train_pinyin_filter_format.txt
DEV_DATA_PATH=$FORMAT_DATA_DIR/aishell_test.txt
TEST_DATA_PATH=$FORMAT_DATA_DIR/magic_test.txt

# 新增拼音token文件
PINYIN_TOKEN_PATH="/ssd1/users/fangzheng/data/mt_error/source_data/dict/zh_cidian/pinyin_token.txt"

# 日志
LOG_FILE=$LOG_DIR/train_compare_complete_pinyin_log.txt

# max_input_len=48
nohup python -u run_model/run_complete_pinyin.py \
  --do_train \
  --pretrain_model_path=$PRE_TRAINED_MODEL_DIR \
  --output_dir=$FINE_TUNING_MODEL_DIR \
  --model_save_path=$FINETUNE_MODEL_PATH \
  --logging_dir=$LOG_DIR \
  --pinyin_token_path=$PINYIN_TOKEN_PATH \
  --train_data_path=$TRAIN_DATA_PATH \
  --dev_data_path=$DEV_DATA_PATH \
  --test_data_path=$TEST_DATA_PATH \
  --dataloader_proc_num=8 \
  --num_train_epochs=10 \
  --per_device_train_batch_size=90 \
  --per_device_eval_batch_size=500 \
  --eval_batch_step=5000 \
  --require_improvement=80000 \
  --max_input_len=40 \
  --correct_label_num=21537 \
  --learning_rate=5e-5 \
  --weight_decay=0.01 \
  --warmup_ratio=0.1 \
  --seed=42 \
  > $LOG_FILE 2>&1 &