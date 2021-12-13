export CUDA_VISIBLE_DEVICES=5

# 根路径
ROOT_DIR="/ssd1/users/fangzheng/data/mt_error"
# 格式化数据路径
DATA_DIR=$ROOT_DIR/fairseq_format

# levt transformer模型存储路径
LEVT_TRANSFORMER_MODEL_PATH=$ROOT_DIR/model/transformer/fairseq_levt_transformer

LOG_DIR=log
mkdir -p $LOG_DIR

# 数据预处理
# LOG_FILE=$LOG_DIR/fair_preprocess
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
# LEVT_TRANSFORMER_TRAIN_LOG_FILE=$LOG_DIR/fairseq_levt_transformer_bstc_train_log.txt
# nohup python -u /ssd1/users/fangzheng/project/fairseq-0.10.2/fairseq_cli/train.py \
#     $DATA_DIR/data-bin \
#     --arch levenshtein_transformer \
#     --task translation_lev \
#     --ddp-backend=no_c10d \
#     --source-lang zh \
#     --target-lang en \
#     --max-epoch 30 \
#     --fp16 \
#     --noise random_delete \
#     --share-all-embeddings \
#     --optimizer adam \
#     --adam-betas '(0.9, 0.98)' \
#     --clip-norm 0.0 \
#     --lr 5e-4 \
#     --warmup-init-lr '1e-07' \
#     --lr-scheduler inverse_sqrt \
#     --warmup-updates 10000 \
#     --dropout 0.3 \
#     --weight-decay 0.01 \
#     --criterion nat_loss \
#     --label-smoothing 0.1 \
#     --decoder-learned-pos \
#     --encoder-learned-pos \
#     --apply-bert-init \
#     --save-dir $LEVT_TRANSFORMER_MODEL_PATH \
#     --num-workers 4 \
#     --batch-size 384 \
#     --keep-best-checkpoints 1 \
#     --save-interval-updates 3000 \
#     --no-progress-bar \
#     --log-interval 100 \
#     > $LEVT_TRANSFORMER_TRAIN_LOG_FILE 2>&1 &

# 模型预测
LEVT_TRANSFORMER_TEST_LOG_FILE=$LOG_DIR/fairseq_levt_transformer_test_log2.txt

nohup python -u /ssd1/users/fangzheng/project/fairseq-0.10.2/fairseq_cli/generate.py \
    $DATA_DIR/data-bin \
    --task translation_lev \
    --iter-decode-max-iter 1 \
    --iter-decode-eos-penalty 0 \
    --gen-subset test \
    --fp16 \
    --batch-size 500 \
    --beam 1 \
    --no-progress-bar \
    --path $LEVT_TRANSFORMER_MODEL_PATH/checkpoint_last.pt \
    > $LEVT_TRANSFORMER_TEST_LOG_FILE 2>&1 &
