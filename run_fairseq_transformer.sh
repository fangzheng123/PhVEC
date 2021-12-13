export CUDA_VISIBLE_DEVICES=7

# 根路径
ROOT_DIR="/ssd1/users/fangzheng/data/mt_error"
# 格式化数据路径
DATA_DIR=$ROOT_DIR/fairseq_format

# transformer模型存储路径
TRANSFORMER_MODEL_PATH=$ROOT_DIR/model/transformer/fairseq_transformer

LOG_DIR=log
mkdir -p $LOG_DIR
LOG_FILE=$LOG_DIR/fair_preprocess

# 数据预处理
# nohup python -u /ssd1/users/fangzheng/project/fairseq-0.10.2/fairseq_cli/preprocess.py \
#     --source-lang zh \
#     --target-lang en \
#     --joined-dictionary \
#     --trainpref $DATA_DIR/train \
#     --validpref $DATA_DIR/valid \
#     --testpref $DATA_DIR/test \
#     --destdir $DATA_DIR/data-bin \
#     --workers 8 \
#     > $LOG_FILE 2>&1 &

# 模型训练
# TRANSFORMER_TRAIN_LOG_FILE=$LOG_DIR/fairseq_transformer_bstc_train_log.txt
# nohup python -u /ssd1/users/fangzheng/project/fairseq-0.10.2/fairseq_cli/train.py \
#     $DATA_DIR/data-bin \
#     --arch transformer \
#     --source-lang zh \
#     --target-lang en \
#     --max-epoch 30 \
#     --fp16 \
#     --scoring wer \
#     --share-decoder-input-output-embed \
#     --optimizer adam \
#     --adam-betas '(0.9, 0.98)' \
#     --clip-norm 0.0 \
#     --lr 5e-4 \
#     --lr-scheduler inverse_sqrt \
#     --warmup-updates 4000 \
#     --dropout 0.3 \
#     --weight-decay 0.0001 \
#     --criterion label_smoothed_cross_entropy \
#     --label-smoothing 0.1 \
#     --save-dir $TRANSFORMER_MODEL_PATH \
#     --num-workers 4 \
#     --batch-size 512 \
#     --report-accuracy \
#     --keep-best-checkpoints 3 \
#     --save-interval-updates 2000 \
#     --no-progress-bar \
#     --log-interval 100 \
#     > $TRANSFORMER_TRAIN_LOG_FILE 2>&1 &

# 模型预测
TRANSFORMER_TEST_LOG_FILE=$LOG_DIR/fairseq_transformer_test_log.txt

nohup python -u /ssd1/users/fangzheng/project/fairseq-0.10.2/fairseq_cli/generate.py \
    $DATA_DIR/data-bin \
    --gen-subset test \
    --fp16 \
    --batch-size 500 \
    --beam 1 \
    --no-progress-bar \
    --path $TRANSFORMER_MODEL_PATH/checkpoint_last.pt \
    > $TRANSFORMER_TEST_LOG_FILE 2>&1 &
