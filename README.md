# ASR纠错

## 简介
该工具用于ASR纠错，包括对原始语音文件的识别以及基于序列标注及生成模型的文本纠错

## 依赖环境
* python 3.7及以上
* torch 1.8.0
* transformers 4.5.1
* espnet 0.9.9
* fairseq 0.10.2

## 数据准备
* 指定领域纯文本语料（按行分隔，每行一条纯文本）
* 指定领域实体词典 （每行为 实体类别\t实体名称）
* 预训练BERT模型 （推荐使用哈工大WWM模型）
* 词向量文件（用于短语挖掘中特征计算，推荐腾讯词向量）
* 停用词文件 (推荐百度停用词表)
* libcut及pos工具文件置于tool目录下即可
* 相关数据可从hdfs上下载，目录：/user/fangzheng.1125/share_data下载

## 挖掘流程
1. 挖掘领域高质量短语，并扩充种子实体字典
    - 使用PhraseMining模型挖掘领域高质量短语
    - 根据中心词挖掘短语中的领域实体
    - 此步骤输出**格式化文本，领域短语，新增实体**
2. 人工审核从短语挖掘出的新实体
    - 审核第1步新增的实体（大概10分钟能够审核完毕）
    - 此步骤输出**清洗过的短语实体字典** (特别注意: 数据格式保持与种子实体文件格式一致)
3. 远程打标数据，训练模型
    - 根据实体字典，通过字典树匹配得到文本中的实体及位置 （只有出现实体的文本才会用于训练）
    - 通过分词去除部分噪声标注语料
    - 构建BERT Softmax模型训练语料
    - 构建BERT AutoNER模型训练语料（相对于BERT Softmax模型，新增unknown实体（高质量短语）打标）
    - 训练BERT Softmax 及 BERT AutoNER模型
    - 此步骤输出**训练好的BERT Softmax 及 BERT AutoNER模型文件**
4. 使用训练好的模型测试全量文本
    - 使用BERT Softmax模型测试全量文本
    - 使用BERT AutoNER模型测试全量文本
    - 此步骤输出**BERT Softmax 及 BERT AutoNER模型的预测结果**
5. 根据模型预测结果挖掘新实体，并对新实体打分
    - 挖掘BERT Softmax及BERT AutoNER 预测的新实体
    - 结合BERT Softmax，BERT AutoNER，PhraseMining模型（**为第1步训练好的模型**）给每个实体进行置信度打分
    - 此步骤输出**排序后的新实体打分文件**
6. 人工审核打分高的新实体
    - 人工打标头部分数高的新实体（审核时间视实体数量而定，头部4500个实体需要2个小时打标）
    - 此步骤输出**新实体头部部分的干净实体文件**（
    打标后实体数据格式为: 实体名\t实体类别;名称命名为manual_pos_label_$ITER_VERSION_INDEX）
7. 将审核后的新实体加入实体字典，远程打标下一轮训练数据
    - 根据人工审核结果构建新的正例实体字典及负例实体字典
    - 通过远程监督构建训练数据（包含正例及负例）
    - 通过分词去除部分噪声标注语料
    - 此步骤输出**扩充后的正例实体字典，远程打标的下一轮训练文件**
8. 重复3-7步直至新挖掘出的实体满足设定数量
9. 基于挖掘出的新实体，构建远程打标数据，并训练蒸馏后的CNN模型(或直接训练CNN模型)


其中**第2步及第6步**需要人工干预（审核新挖掘的实体是否包含噪声）

## 工具产出
* 领域实体字典
* 实体识别模型（训练好的BERT Softmax模型可直接用于实体识别）

## 工具使用
### 1. 挖掘领域高质量短语，并扩充种子实体字典
```shell
sh a_run_seed_expand.sh
```

```text
参数说明
├── 输入
|  └── TASK_NAME                        # 当前任务领域
|  └── SOURCE_DIR                       # 所有数据存储根路径
|  └── TASK_DIR                         # 当前任务根路径
|  └── SOURCE_DATA_PATH                 # 待挖掘的自由文本数据
|  └── SEED_ENTITY_PATH                 # 种子实体文件
|  └── WORD_VEC_PATH                    # 词向量文件
|  └── STOPWORD_PATH                    # 停用词
|  └── SYMBOL_PATH                      # 词性标注中所有标点符号
|  └── POS_LABEL_PATH                   # 词性标注所有标签
|  └── MAX_PHRASE_LEN                   # 短语最大长度(词语级别)
|  └── MIN_PHRASE_FREQ                  # 候选短语最小频次（少于此频次短语将被丢弃）
|  └── BASE_MODEL_NUM                   # 基分类器个数 (随机森立中基分类器数)
|  └── NEG_POS_TIMES                    # 短语挖掘模型中训练集中负例数与正例数之比
|  └── HEADWORD_MIN_FREQ                # 中心词出现的最小频次 
|  └── HEAD_PHRASE_NUM                  # 根据中心词挖掘实体时所选取的头部短语数
├── 输出
|  └── PHRASE_MODEL_PATH                # 短语模型存储路径
|  └── PHRASE_PRED_RESULT_PATH          # 挖掘出的领域短语存储路径
|  └── CANDIDATE_PHRASE_ENTITY_PATH     # 根据中心词从短语中选取的候选实体存储路径
|  └── TEXT_FORMAT_PATH                 # 格式化自由文本数据（后续数据处理将复用）

除文件路径外，其余参数均可使用默认值
```

### 2. 远程打标数据，训练模型
```shell
sh b_run_model_train.sh
```

```text
参数说明
├── 远程打标参数
|  └── CUDA_VISIBLE_DEVICES_SOFT                    # BERT Softmax模型使用GPU
|  └── CUDA_VISIBLE_DEVICES_AUTO                    # BERT AutoNER模型使用GPU
|  └── TASK_NAME                                    # 当前任务领域
|  └── ITER_VERSION_INDEX                           # 模型迭代轮数
|  └── DATA_ROOT_DIR                                # 所有数据存储根路径
|  └── TASK_DATA_DIR                                # 当前任务根路径
|  └── MODEL_TYPE                                   # BERT预训练模型类型(如bert, Robert, bert wwm等)
|  └── PRE_TRAINED_MODEL_DIR                        # 预训练模型存储路径
|  └── LABELS                                       # 要训练实体的类别（类别名称通过','连接）
|  └── SEED_ENTITY_PATH                             # 种子实体文件 
|  └── PHRASE_PRED_RESULT_PATH                      # 挖掘出的短语文件
|  └── PHRASE_ENTITY_CLEAN_PATH                     # 人工审核过的短语实体文件
|  └── TEXT_FORMAT_PATH                             # 格式化的自由文本数据
|  └── TRAIN_SOFT_DATA_PATH                         # BERT Softmax模型训练文件存储路径
|  └── DEV_SOFT_DATA_PATH                           # BERT Softmax模型验证文件存储路径
|  └── TRAIN_AUTO_DATA_PATH                         # BERT AutoNER模型训练文件存储路径
|  └── DEV_AUTO_DATA_PATH                           # BERT AutoNER模型验证文件存储路径

├── BERT Softmax模型训练参数
|  └── loss_type                                    # 模式使用损失函数类型 {ce: CrossEntropyLoss, lsr:LabelSmoothingCrossEntropy, focal:FocalLoss}
|  └── require_improvement                          # 间隔一定batch数模型性能不提升，则early stop
|  └── max_seq_length                               # 文本最大长度 (token级别)
|  └── per_eval_batch_step                          # 每隔多少batch在验证集上验证模型效果
|  └── per_gpu_train_batch_size                     # 每块GPU上训练模型时batch数
|  └── per_gpu_dev_batch_size                       # 每块GPU上验证模型时batch数
|  └── num_train_epochs                             # 训练模型轮数

├── BERT AutoNER模型训练参数
|  └── do_no_seg                                    # 模型不分词（基于token间的连接关系）
|  └── loss_type                                    # 同上
|  └── require_improvement                          # 同上
|  └── max_seq_length                               # 同上 
|  └── seq_max_word_num                             # 文本最多包含的词语数 (词语级别，do_no_seg设置时此设置无效)
|  └── per_eval_batch_step                          # 同上
|  └── per_gpu_train_batch_size                     # 同上
|  └── per_gpu_dev_batch_size                       # 同上
|  └── num_train_epochs                             # 同上

文件路径需要指定，模型的训练及验证batch数需要根据GPU内存大小进行调整，大多数参数均可使用默认值
```

### 3. 使用训练好的模型测试全量文本
```shell
sh c_run_model_pred.sh
```

```text
参数说明
├── BERT Softmax模型预测参数
|  └── task_name                                    # 同上
|  └── iter_version_index                           # 模型迭代轮数
|  └── gpu_devices                                  # 测试模型使用GPU
|  └── pre_trained_model_path                       # 同训练步骤
|  └── model_type                                   # 同训练步骤
|  └── model_dir                                    # 微调后模型存储路径
|  └── pred_data_path                               # 预测数据存储路径
|  └── output_path                                  # 预测结果存储路径
|  └── loss_type                                    # 同训练步骤
|  └── label_names                                  # 同训练步骤
|  └── max_seq_length                               # 同训练步骤
|  └── test_batch_size                              # 每块GPU上测试模型时batch数

├── BERT AutoNER模型预测参数
|  └── do_no_seg                                    # 模型不分词（基于token间的连接关系）
|  └── task_name                                    # 同上
|  └── iter_version_index                           # 模型迭代轮数
|  └── gpu_devices                                  # 测试模型使用GPU
|  └── pre_trained_model_path                       # 同训练步骤
|  └── model_type                                   # 同训练步骤
|  └── model_dir                                    # 微调后模型存储路径
|  └── pred_data_path                               # 预测数据存储路径
|  └── output_path                                  # 预测结果存储路径
|  └── loss_type                                    # 同训练步骤
|  └── label_names                                  # 同训练步骤
|  └── max_seq_length                               # 同训练步骤
|  └── seq_max_word_num                             # 同训练步骤
|  └── test_batch_size                              # 每块GPU上测试模型时batch数

文件路径需要指定，模型的测试batch数需要根据GPU内存大小进行调整，大多数参数均可使用默认值
```

### 4. 根据模型预测结果挖掘新实体，并对新实体打分
```shell
sh d_run_entity_extract.sh
```

```text
参数说明
├── 输入
|  └── TASK_NAME                                    # 当前任务领域
|  └── iter_version_index                           # 模型迭代轮数
|  └── SEED_EXPAND_ENTITY_PATH                      # 第一步扩充后的种子实体文件
|  └── PRED_SOFT_RESULT_PATH                        # BERT Softmax预测结果文件
|  └── PRED_AUTO_RESULT_PATH                        # BERT AutoNER预测结果文件
|  └── WORD_VEC_PATH                                # 同步骤1
|  └── SYMBOL_PATH                                  # 同步骤1
|  └── POS_LABEL_PATH                               # 同步骤1
|  └── STOPWORD_PATH                                # 同步骤1
|  └── PHRASE_MODEL_PATH                            # 同步骤1（第1步训练好的PhraseMining模型）
|  └── PHRASE_INTER_DATA_DIR                        # 同步骤1
|  └── MAX_PHRASE_LEN                               # 同步骤1
|  └── MIN_PHRASE_FREQ                              # 同步骤1
|  └── BASE_MODEL_NUM                               # 同步骤1
|  └── NEG_POS_TIMES                                # 同步骤1
├── 输出
|  └── ENTITY_MULTI_SCORE_RANK_PATH                 # 结合多个模型打分排序后的新实体文件

除文件路径外，其余参数均可使用默认值，且短语相关参数保持与步骤1一致
```

### 5. 将审核后的新实体加入实体字典，远程打标下一轮训练数据
```shell
sh e_run_next_label.sh
```

```text
参数说明
├── 输入
|  └── TASK_NAME                                    # 当前任务领域
|  └── iter_version_index                           # 模型迭代轮数
|  └── SOURCE_DIR                                   # 所有数据存储根路径
|  └── TASK_DIR                                     # 当前任务根路径
|  └── SEED_EXPAND_ENTITY_PATH                      # 第1步扩充后的种子实体文件
|  └── RANK_NEW_ENTITY_PATH                         # 第4步生成的排序多模型打分文件
|  └── MANUAL_NEG_LABEL_PATH                        # 人工在排序头部新实体中标注的负例
|  └── LABEL_HEAD_NUM                               # 人工打标头部正例数
|  └── NEG_TAIL_NUM                                 # 从新实体排序尾部筛选负例的数量
|  └── TEXT_FORMAT_PATH                             # 第1步格式化的自由文本数据
├── 输出
|  └── TRAIN_NEXT_DATA_PATH                         # 下一轮远程标注训练数据（候选BERT Softmax模型及BERT AutoNER模型将使用相同训练数据）
|  └── DEV_NEXT_DATA_PATH                           # 下一轮远程标注验证数据（候选BERT Softmax模型及BERT AutoNER模型将使用相同验证数据）
|  └── ENTITY_ADD_EXTRACT_PATH                      # 当前轮挖掘出的新实体（包括种子实体）

除文件路径外，其余参数均可使用默认值
```

### 6. 重复3-5步
在迭代运行时只需要修改脚本中的iter_version_index参数，每轮+1即可，其余参数均可不修改。

### 7. 模型蒸馏
```shell
sh f_run_distill.sh
```

```text
参数说明
├── 输入
|  └── do_single                                    # 是否单独训练学生模型(无蒸馏过程)
|  └── TASK_NAME                                    # 当前任务领域
|  └── iter_version_index                           # 经过多少轮迭代获取当前新实体
|  └── TASK_DIR                                     # 当前任务根路径
|  └── student_type                                 # 学生模型类型，cnn或gru
|  └── T                                            # 蒸馏温度
|  └── cnn_num_filters                              # cnn模型卷积核数量

除文件路径外，其余参数均可使用默认值
```

## 代码说明
```text
├── data_process　                           # 实体数据处理
|  └── clean_crawl_data.py      
|  └── entity_combine.py
|  └── entity_feature.py
|  └── entity_filter.py
|  └── entity_label.py
|  └── headword_expand.py

├── model　                                  # 模型
|  └── model_config                         # 模型配置 
|  | └── base_model_config.py
|  | └── bert_autoner_config.py
|  | └── bert_softmax_config.py
|  └── model_data_process                   # 模型数据处理
|  | └── bert_autoner_data_processor.py
|  | └── bert_softmax_data_processor.py
|  └── model_define                         # 模型结构定义
|  | └── bert_autoner.py
|  | └── bert_softmax.py
|  | └── loss.py
|  └── model_metric                         # 模型评估模块
|  | └── bert_autoner_metric.py
|  | └── bert_softmax_metric.py
|  └── model_process                        # 模型处理，包括训练、验证、测试
|  | └── bert_autoner_process.py
|  | └── bert_softmax_process.py

├── ner_service　                            # NER服务
|  └── bert_ner_controller.py      
|  └── ner_config.py

├── phrase_mining　                          # 短语挖掘
|  └── data_util.py      
|  └── entity_processor.py
|  └── phrase_config.py      
|  └── phrase_controller.py
|  └── phrase_feature.py      
|  └── phrase_processor.py
|  └── text_processor.py
|  └── xgb_forest.py

├── run_model　                              # 运行相关模块
|  └── run_bert_autoner.py      
|  └── run_data_label.py
|  └── run_entity_extract.py      
|  └── run_ner_softmax.py
|  └── run_next_data_label.py      
|  └── run_seed_expand.py

├── tool　                                   # 项目使用的外部工具
|  └── libcut      
|  └── pos
|  └── tool.load.py

├── util　                                   # 通用工具类
|  └── arg_util.py      
|  └── data_util.py
|  └── entity_util.py
|  └── file_util.py      
|  └── log_util.py
|  └── model_util.py
|  └── text_util.py
|  └── trie.py    
```
