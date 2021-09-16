
export CUDA_VISIBLE_DEVICES=7


raw_dir="/ssd1/users/fangzheng/data/asr_data/magic_data/wav"  
data_type="test"

# 将数据均分到多个文件夹中
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

# 多个进程处理数据
# asr_dir="/ssd1/users/fangzheng/data/asr_data/magic_data/asr"
# result_dir=$asr_dir/$data_type/
# mkdir -p $result_dir
# mkdir -p log

# for element in `ls $raw_dir | grep ${data_type}_`
# do
#     echo $raw_dir/$element
#     nohup /opt/compiler/gcc-8.2/lib/ld-linux-x86-64.so.2 --library-path \
#     /opt/compiler/gcc-8.2/lib:/ssd1/users/fangzheng/anaconda3/lib:/usr/lib64:$LD_LIBRARY_PATH \
#     /ssd1/users/fangzheng/anaconda3/bin/python -u asr_process.py \
#     --wav_data_dir=$raw_dir/$element \
#     --asr_result_dir=$result_dir \
#     > log/$element 2>&1 &
# done

# for element in {a,b,c,d,e,f}
# do
#     echo $raw_dir/$element
#     nohup /opt/compiler/gcc-8.2/lib/ld-linux-x86-64.so.2 --library-path \
#     /opt/compiler/gcc-8.2/lib:/ssd1/users/fangzheng/anaconda3/lib:/usr/lib64:$LD_LIBRARY_PATH \
#     /ssd1/users/fangzheng/anaconda3/bin/python -u asr_process.py \
#     --wav_data_dir=$raw_dir/$element \
#     --asr_result_dir=$result_dir \
#     > log/$element 2>&1 &
# done
    