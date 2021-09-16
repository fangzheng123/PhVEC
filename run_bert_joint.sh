
export CUDA_VISIBLE_DEVICES=7

# 根路径
ROOT_DIR="/ssd1/users/fangzheng/data/mt_error"

# 预训练模型类型
MODEL_TYPE="BERT-wwm-ext"

# 预训练模型路径
PRE_TRAINED_MODEL_DIR=$ROOT_DIR/pretrain_model/${MODEL_TYPE}/
# 微调模型存储路径
FINETUNE_MODEL_DIR=$ROOT_DIR/model/bert_joint_model
FINETUNE_MODEL_PATH=$FINETUNE_MODEL_DIR/bert_joint_200.pk

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
TEST_DATA_PATH=$FORMAT_DATA_DIR/aishell_test.txt

# 日志
LOG_FILE=$LOG_DIR/bert_joint_train.txt

# 使用DDP, 16GB显存不够无法启动
# /opt/compiler/gcc-4.8.2/lib/ld-linux-x86-64.so.2 --library-path /opt/compiler/gcc-4.8.2/lib \
# /ssd1/users/fangzheng/3.6.5/bin/python -m torch.distributed.launch --nproc_per_node 1 ./run_model/run_bart_correction.py \ 

# max_input_len=48
nohup /opt/compiler/gcc-8.2/lib/ld-linux-x86-64.so.2 --library-path \
/opt/compiler/gcc-8.2/lib:/ssd1/users/fangzheng/anaconda3/lib:/usr/lib64:$LD_LIBRARY_PATH \
/ssd1/users/fangzheng/anaconda3/bin/python -u run_model/run_bert_joint_correction.py \
  --do_train \
  --pretrain_model_path=$PRE_TRAINED_MODEL_DIR \
  --output_dir=$FINE_TUNING_MODEL_DIR \
  --model_save_path=$FINETUNE_MODEL_PATH \
  --token_embed_path=$FORMAT_DATA_DIR/finetune_token_embed.txt \
  --logging_dir=$LOG_DIR \
  --train_data_path=$TRAIN_DATA_PATH \
  --dev_data_path=$DEV_DATA_PATH \
  --test_data_path=$TEST_DATA_PATH \
  --dataloader_proc_num=8 \
  --num_train_epochs=10 \
  --per_device_train_batch_size=90 \
  --per_device_eval_batch_size=500 \
  --eval_batch_step=5000 \
  --require_improvement=80000 \
  --max_input_len=48 \
  --learning_rate=5e-5 \
  --weight_decay=0.01 \
  --warmup_ratio=0.1 \
  --seed=42 \
  > $LOG_FILE 2>&1 &