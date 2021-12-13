
export CUDA_VISIBLE_DEVICES=7

# wav文件夹绝对路径
raw_dir="/ssd1/users/fangzheng/data/asr_data/magic_data/wav" 
# 数据类型，train/dev/test 
data_type="test"

# 将数据均分到多个文件夹中，从而能够并行对多个不同文件夹下的语音文件进行解码(由于此处使用GPU, 因此不宜创建太多文件夹)
moved=0
target=""
for element in `ls $raw_dir/$data_type`
do
    if [ `expr $moved % 5` = 0 ]; then
        target=$raw_dir/${data_type}_1
    elif [ `expr $moved % 5` = 1 ];then
        target=$raw_dir/${data_type}_2
    elif [ `expr $moved % 5` = 2 ];then
        target=$raw_dir/${data_type}_3
    elif [ `expr $moved % 5` = 3 ];then
        target=$raw_dir/${data_type}_4
    elif [ `expr $moved % 5` = 4 ];then
        target=$raw_dir/${data_type}_5
    fi
    echo $moved
    mkdir -p $target
    cd $raw_dir/$data_type
    cp -r $element $target/
    moved=`expr $moved + 1`  
done


# 多个进程解码多个文件夹下的数据(对上述拆分的文件夹分别解码)
asr_dir="/ssd1/users/fangzheng/data/asr_data/magic_data/asr"

# 预训练ASR模型, ESPNET工具包中提供
PRE_TRAIN_ASR_MODEL_PATH="/ssd1/users/fangzheng/data/asr_data/pretrian_model/asr_train_asr_conformer3_raw_char_batch_bins4000000_accum_grad4_sp_valid.acc.ave.zip"
CACHE_DIR="/ssd1/users/fangzheng/data/asr_data/pretrian_model/espnet"

# 解码结果文件夹，每个文档的asr结果将存储到一个json文件中
result_dir=$asr_dir/$data_type/

mkdir -p $result_dir
mkdir -p log

for element in `ls $raw_dir | grep ${data_type}_`
do
    echo $raw_dir/$element
    nohup python -u espnet/asr_process.py \
    --pre_train_model_path=$PRE_TRAIN_ASR_MODEL_PATH \
    --cache_dir=$CACHE_DIR \
    --wav_data_dir=$raw_dir/$element \
    --asr_result_dir=$result_dir \
    > log/$element 2>&1 &
done

# primeword数据中有的文件夹还有a,b,c等字母命名
# for element in {a,b,c,d,e,f}
# do
#     echo $raw_dir/$element
#     nohup python -u espnet/asr_process.py \
#     --wav_data_dir=$raw_dir/$element \
#     --asr_result_dir=$result_dir \
#     > log/$element 2>&1 &
# done
    