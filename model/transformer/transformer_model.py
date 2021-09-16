#encoding: utf-8
'''
@File    :   transformer_model.py
@Time    :   2021/04/27 14:29:45
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import math

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import (
    TransformerEncoder,
    TransformerDecoder,
    TransformerEncoderLayer,
    TransformerDecoderLayer
)

class Seq2SeqTransformer(nn.Module):
    """
    Transformer模型
    """
    def __init__(self, args):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model=args.emb_size, nhead=args.head, dim_feedforward=args.dim_feedforward
        )
        decoder_layer = TransformerDecoderLayer(
            d_model=args.emb_size, nhead=args.head, dim_feedforward=args.dim_feedforward
        )

        # 输入与输出共享相同TokenEmbedding
        self.token_embedding = TokenEmbedding(args.src_vocab_size, args.emb_size)
        self.positional_encoding = PositionalEncoding(args.emb_size, dropout=args.dropout)
        
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=args.encoder_layer_num)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=args.decoder_layer_num)
        self.generator = nn.Linear(args.emb_size, args.tgt_vocab_size)
    
    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor, \
        src_padding_mask: Tensor, tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):

        # tgt输入中不包含最后一个字符
        tgt = tgt[:, :-1]

        # 将Batch维度置于第二维
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        # 获取token表示
        src_emb = self.positional_encoding(self.token_embedding(src))
        tgt_emb = self.positional_encoding(self.token_embedding(tgt))

        # 获取encoder输出
        memory = self.transformer_encoder(src_emb, src_mask[0], src_padding_mask)

        # 获取decoder输出
        outs = self.transformer_decoder(tgt_emb, memory, tgt_mask[0], None, tgt_padding_mask, memory_key_padding_mask)

        # 将Batch维度置于第一维
        outs = outs.transpose(0, 1)

        return self.generator(outs)
    
    def encode(self, src: Tensor, src_mask: Tensor, src_padding_mask: Tensor):
        return self.transformer_encoder(self.positional_encoding(self.token_embedding(src)), src_mask[0], src_padding_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor, memory_key_padding_mask: Tensor):
        return self.transformer_decoder(
            self.positional_encoding(self.token_embedding(tgt)), memory, 
            tgt_mask[0], memory_key_padding_mask=memory_key_padding_mask)

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    
    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super().__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        # 注册为buffer, 而不是parameter, 训练时无需更新此参数
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0),:])



    
         
        
