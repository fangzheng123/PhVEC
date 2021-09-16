# ASR纠错

## 简介
该工具用于ASR纠错，包括对原始语音文件的识别以及基于序列标注及生成模型的文本纠错，主要支持的模型包括基于BERT,BERT+CTC,PhVEC,Transformer,mBART的文本纠错

## 依赖环境
* python 3.7及以上
* torch 1.8.0
* transformers 4.5.1
* espnet 0.9.9
* fairseq 0.10.2
* 其他依赖具体见requirements.txt

## 相关准备
* ASR识别结果及Transcript (每一条为json格式数据, 如{"asr":"", "transcript":""}, 示例中字段必须包含，其他字段可自行添加)
* 预训练BERT模型 （推荐使用Chinese-BERT-wwm模型, github搜索下载即可）

## 测试数据构建
1. ASR语音识别 (若已经有ASR与Transcript对齐数据，可直接跳过)
    - 将文件夹下的.wav语音文件转化为文本
    
    ```shell
    sh run_asr.sh
    ```

    - 其中ASR识别主要分为2步，第一步将所有文档(一个文档包含多个句子，每个wav文件对应一条句子的语音数据)均分到多个文件夹下，这样后续可以并行解码；第二步则对每个文件夹下的文档使用GPU进行解码(此处我们直接使用ESPNET提供的在AISHELL训练数据上预训练得到的中文ASR模型)

    ```text
    参数说明
    ├── 输入
    |  └── pre_train_model_path             # ESPNET中下载的预训练模型
    |  └── cache_dir                        # ESPNET中下载的预训练模型
    |  └── wav_data_dir                     # 当前wav文件根路径
    
    ├── 输出
    |  └── asr_result_dir                   # 输出根路径

    每个文档的asr结果将以json文件格式存储到asr_result_dir路径下
    ```
    
2. 对齐ASR识别结果与Transcript, 构建格式化测试数据
    ```shell
    cd espnet
    python data_align.py
    ```

    ```text
    参数说明
    ├── 输入
    |  └── asr_dir                           # 第一步中ASR中识别路径
    |  └── transcript_data_path              # Transcirpt文件
    
    ├── 输出
    |  └── asr_format_data_path              # 输出文件路径

    输出为一个单独的格式化文件，每一行为json格式数据, 如{"doc_id":"", "sent_id":"", "asr":"", "transcript":""}
    ```

## 伪训练数据构建
1. 分析ASR中错误，构建混淆集合
```shell
cd data_process
python asr_error_process.py
```

2. 根据混淆集合多进程生成含错误数据
```shell
cd data_process
python pseudo_error_generate.py (推荐nohup运行, 可根据数量量调整进程数)
```

3. 生成错误汉字对应拼音并对齐拼音字母汉字，生成训练格式化数据
```shell
cd data_process
python pseudo_data_format.py (推荐nohup运行)
```

```text
包含错误的格式化数据:
{"asr": "之后陈奕迅充当厨师考肉给歌迷吃", "transcript": "之后陈奕迅充当厨师烤肉给歌迷吃", "errors": [{"error_word": "考", "label_word": "烤", "error_pinyin": "kao", "label_pinyin": "kao", "detect_error_range": [9, 10], "correct_input": "之后陈奕迅充当厨师
考 k a o 肉给歌迷吃", "correct_label": "之后陈奕迅充当厨师烤 烤 烤 烤 肉给歌迷吃"}]}

不含错误格式化数据:
{"asr": "知道你的想法", "transcript": "知道你的想法", "errors": []}
```




## 模型训练及测试
### 1.PhVEC模型
```shell
sh run_bert_joint.sh
```

```text
重点参数说明
├── 模型参数
|  └── do_train/do_eval                             # 训练/测试
|  └── pretrain_model_path                          # 预训练BERT模型存储路径
|  └── output_dir                                   # 训练模型存储文件夹
|  └── model_save_path                              # 最优训练模型存储路径
|  └── token_embed_path                             # 最后一层embedding存储路径，实验分析用
|  └── logging_dir                                  # 日志文件

├── 配置参数
|  └── dataloader_proc_num                          # 数据并行加载进程数
|  └── num_train_epochs                             # 总训练epoch数
|  └── eval_batch_step                              # 每隔多少步在验证数据集上验证
|  └── require_improvement                          # 超过配置数early stop
|  └── max_input_len                                # 文本最大输入长度，多截少padding

文件路径需要指定，模型的训练及验证batch数需要根据GPU内存大小进行调整，大多数参数均可使用默认值
```

### 2.BERT模型
```shell
sh run_bert_single.sh
```

```text
模型参数同1, 由于此处直接进行纠错，因此batch_size可设置更大
```


### 3.BERT+CTC模型
```shell
sh run_bert_single.sh
```

```text
重点参数说明
├── 模型参数
|  └── do_ctc                                       # BERT Softmax模型使用GPU

├── 配置参数
|  └── max_input_len                                # 最大输入长度（设置为输出长度2倍！！）
|  └── max_target_len                               # 最大输出长度（保持和模型1和2中的max_input_len相同即可）

文件路径需要指定，模型的训练及验证batch数需要根据GPU内存大小进行调整，大多数参数均可使用默认值
```

### 4.Transformer模型
```shell
sh run_fairseq_transformer.sh
```

```text
参数均可使用默认值，具体配置可参考fairseq
```

fairseq中数据预处理需要设定为其框架要求格式，此处首先要对语料进行BPE分词处理，为了其他模型保持一致，我们此处使用BERT Tokenizer分词结果，然后以空格分隔分词结果，其中数据预处理为:
```shell
cd data_process
python fairseq_process.py
sh run_fairseq_transformer.sh (运行fairseq_cli/preprocess.py模块即可)
```

### 5.Levt Transformer模型
```shell
sh run_fairseq_levt_transformer.sh
```

```text
参数均可使用默认值，具体配置可参考fairseq，数据预处理同模型4
```

### 6.mBART模型
```shell
sh run_bart.sh
```

```text
模型参数同1, 由于mBART是12层Encoder+12层Decoder，16GB GPU batch_size最大设置不超过4
```

### 7.BERT+完整拼音token模型
```shell
sh run_complete_pinyin.sh
```

```text
模型参数同1
```

## 代码总体结构说明
```text
├── data_process　                          # 所有数据处理（清洗过滤，伪数据构造等）

├── espnet                                  # ASR识别处理

├── model                                   # 所有模型
|  └── bert_joint                           # 联合纠错模型（联合训练错误检测及错误纠正模块） 
|  | └── bert_joint_dataloader.py           # 数据并行加载及tokenize
|  | └── bert_joint_model.py                # 模型结构定义
|  | └── bert_joint_process.py              # 模型训练与测试
|  └── bert_joint_pinyin                    # 联合纠错模型（pinyin作为完整token加入纠错模型中）
|  └── bert_pipeline                        # 串行纠错模型（分开训练检测模块和错误纠正模块）
|  └── bert_single                          # 直接预测纠错结果(无错误检测模块)
|  └── mbart                                # 基于mBART的生成式纠错模型

├── other_ignore                            # 结果分析及自己实现的transformer模型等，可忽略


├── run_model　                             # 运行不同算法模块,与.sh脚本直接关联的python文件
|  └── run_bart_correction.py               # 运行mBART模型              
|  └── run_bert_joint_correct.py            # 运行我们的PhVEC模型（也包括bert_joint + CTC模型）
|  └── run_bert_single_correction.py        # 基于BERT模型直接纠错（也包括BERT+CTC模型）
|  └── run_pinyin_compare.py                # 类似我们的PhVEC模型，只是pinyin token作为完整的toke加入到纠错模块中

├── util　                                  # 通用工具类
|  └── arg_util.py                          # 解析模型参数类，新增参数时更改此类代码
|  └── asr_score_util.py                    # 指标评测 cer/wer计算工具类
|  └── file_util.py                         # 文件读取存储工具类
|  └── log_util.py                          # 日志打印工具类
|  └── model_util.py                        # 模型相关工具类
|  └── text_util.py                         # 文本字符串处理工具类
```
