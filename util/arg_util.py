#encoding: utf-8
'''
@File    :   arg_util.py
@Time    :   2021/04/20 20:58:37
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

from typing import Optional
from dataclasses import dataclass, field

from transformers import Seq2SeqTrainingArguments

@dataclass
class BaseArguments(Seq2SeqTrainingArguments):
    """
    基础参数+从Seq2SeqTrainingArguments中继承的大量公共参数
    """
    train_data_path: Optional[str] = field(default=None)
    dev_data_path: Optional[str] = field(default=None)
    test_data_path: Optional[str] = field(default=None)
    model_save_path: Optional[str] = field(default=None)
    dataloader_proc_num: int = field(default=4)
    require_improvement: int = field(default=100)
    eval_batch_step: int = field(default=1000)
    ignore_pad_token_for_loss: bool = field(default=True)
    # train_batch_size: int = field(default=2)
    max_input_len: int = field(default=40)
    max_output_len: int = field(default=32) 
    num_beams: int = field(default=1)
    dropout: float = field(default=0.1)
    
@dataclass
class BARTArguments(BaseArguments):
    """
    BART模型参数
    """
    pretrain_model_path: Optional[str] = field(default=None)
    
@dataclass
class TransformerArguments(BaseArguments):
    """
    Transformer模型参数
    """
    pretrain_model_path: Optional[str] = field(default=None)
    emb_size: int = field(default=512) 
    head: int = field(default=8) 
    dim_feedforward: int = field(default=32) 
    encoder_layer_num: int = field(default=1) 
    decoder_layer_num: int = field(default=1) 
    src_vocab_size: int = field(default=21128)
    tgt_vocab_size: int = field(default=21128)

@dataclass
class BERTArguments(BaseArguments):
    """
    BERT模型参数
    """
    do_split: bool = field(default=False)
    do_ctc: bool = field(default=False)
    pretrain_model_path: Optional[str] = field(default=None)
    max_detect_input_len: int = field(default=64)
    max_correct_input_len: int = field(default=72)
    bert_hidden_size: int = field(default=768)
    detect_label_num: int = field(default=2)
    correct_label_num: int = field(default=21128)
    max_target_len: int = field(default=40)
    token_embed_path: Optional[str] = field(default=None)
    pinyin_token_path: Optional[str] = field(default=None)

@dataclass
class BERTPipelineArguments(BERTArguments):
    """
    BERT模型参数
    """
    do_detect: bool = field(default=False)
    do_correct: bool = field(default=False)
    detect_model_path: Optional[str] = field(default=None)
    correct_model_path: Optional[str] = field(default=None)
    