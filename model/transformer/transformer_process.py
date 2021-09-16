#encoding: utf-8
'''
@File    :   transformer_process.py
@Time    :   2021/04/27 14:30:45
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import torch
import torch.nn as nn
import numpy as np
from accelerate import Accelerator
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup

from util.asr_score_util import ASRScoreUtil
from util.log_util import LogUtil

class TransformerProcess(object):
    """
    Transformer模型处理类
    """
    def __init__(self, args, transformer_model, bert_tokenizer):
        self.args = args
        self.transformer_model = transformer_model
        self.bert_tokenizer = bert_tokenizer
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    def _generate_square_subsequent_mask(self, sz):
        """
        We create a ``subsequent word`` mask to stop a target word from attending to its subsequent words. 
        @param:
        @return:
        """
        mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        return mask 

    def _add_mask(self, batch_data):
        """
        给src,tgt加入mask
        @param:
        @return:
        """
        src_mask = torch.zeros((self.args.max_input_len, self.args.max_input_len)).type(torch.bool)
        tgt_mask = self._generate_square_subsequent_mask(self.args.max_output_len-1)

        # 增加batch维度, 防止多GPU训练时对第一维拆分
        src_mask = src_mask.repeat(len(batch_data), 1, 1)
        tgt_mask = tgt_mask.repeat(len(batch_data), 1, 1)

        batch_data["src_mask"] = src_mask
        batch_data["tgt_mask"] = tgt_mask
        batch_data["memory_key_padding_mask"] = batch_data["src_padding_mask"]

    def train(self, train_loader, dev_loader):
        """
        训练transformer模型
        @param:
        @return:
        """
        self.transformer_model.train()

        for p in self.transformer_model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        optimizer = torch.optim.Adam(self.transformer_model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        
        # 多GPU训练
        if torch.cuda.device_count() > 1:
            self.transformer_model = torch.nn.DataParallel(self.transformer_model)
                    
        # 进行到多少batch
        total_batch = 0
        dev_best_score = 0
        # 上次验证集loss下降的batch数
        last_improve = 0
        # 是否很久没有效果提升
        no_improve_flag = False

        LogUtil.logger.info("Batch Num: {0}".format(len(train_loader)))
        for epoch in range(self.args.num_train_epochs):
            LogUtil.logger.info("Epoch [{}/{}]".format(epoch + 1, self.args.num_train_epochs))
            loss_total = 0
            for step, batch_data in enumerate(train_loader):
                # 加入src_mask及tgt_mask输入
                self._add_mask(batch_data)                
                # 将数据加载到gpu(使用accelerator后可省略)
                batch_data = {k: v.to(self.args.device) for k, v in batch_data.items()}
                seq2seq_output = self.transformer_model(**batch_data)
                optimizer.zero_grad()

                # tgt输出标签不包含第一个字符
                tgt_out = batch_data["tgt"][:, 1:]
                loss = self.loss_fn(seq2seq_output.reshape(-1, seq2seq_output.shape[-1]), tgt_out.reshape(-1))
                loss.backward()
                optimizer.step()
                loss_total += loss.item()
                
                # 输出在训练集和验证集上的效果
                LogUtil.logger.info(step)
                if total_batch % self.args.eval_batch_step == 0 and epoch > 0:
                    dev_loss, dev_bleu_score = self.evaluate(dev_loader)
                    if dev_bleu_score > dev_best_score:
                        dev_best_score = dev_bleu_score
                        torch.save(self.transformer_model.state_dict(), self.args.model_save_path)
                        improve = "*"
                        last_improve = total_batch
                    else:
                        improve = ""
                    msg = "Iter: {0:>3}, Train Loss: {1:>5.2}, Dev Loss: {2:>5.2},  Dev CER score: {3:>5.2} {4}"
                    LogUtil.logger.info(msg.format(total_batch, loss.item(), dev_loss, dev_bleu_score, improve))
                    
                    self.transformer_model.train()

                total_batch += 1
                if total_batch - last_improve > self.args.require_improvement:
                    LogUtil.logger.info("No optimization for a long time, auto-stopping...")
                    no_improve_flag = True
                    break
            if no_improve_flag:
                break

            train_loss = loss_total / len(train_loader)
            LogUtil.logger.info("Train Loss: {0}".format(train_loss))
    
    def evaluate(self, data_loader):
        """
        验证模型
        @param:
        @return:
        """
        self.transformer_model.eval()
        
        loss_total = 0
        all_pred_list = []
        all_label_list = []
        with torch.no_grad():
            for step, batch_data in enumerate(data_loader):
                # 加入src_mask及tgt_mask输入
                self._add_mask(batch_data)                
                # 将数据加载到gpu(使用accelerator后可省略)
                batch_data = {k: v.to(self.args.device) for k, v in batch_data.items()}
                seq2seq_output = self.transformer_model(**batch_data)

                # tgt输出标签不包含第一个字符
                tgt_out = batch_data["tgt"][:, 1:]
                loss = self.loss_fn(seq2seq_output.reshape(-1, seq2seq_output.shape[-1]), tgt_out.reshape(-1))
                loss_total += loss.item()
 
                # src = batch_data["src"][0, :].cpu().numpy().tolist()
                # if self.bert_tokenizer.pad_token_id in src:
                #     src = src[:src.index(self.bert_tokenizer.pad_token_id)]
                #     src.append(self.bert_tokenizer.sep_token_id)
                # src_len = len(src)
                # src = torch.LongTensor(src).unsqueeze(0).to(self.args.device)
                # src_mask = (torch.zeros(src_len, src_len)).type(torch.bool).to(self.args.device)
                # src_mask = src_mask.repeat(len(batch_data), 1, 1)
                # generated_tokens = self.greedy_decode(
                #     src, src_mask, self.args.max_output_len, 
                #     self.bert_tokenizer.cls_token_id, self.bert_tokenizer.sep_token_id)

                generated_tokens = self.batch_greedy_decode(
                    batch_data["src"], batch_data["src_mask"], batch_data["src_padding_mask"],
                    self.args.max_output_len, self.bert_tokenizer.cls_token_id, self.bert_tokenizer.sep_token_id)
                
                generated_tokens = np.array(generated_tokens)
                labels = tgt_out.cpu().numpy()
                
                # decode为字符
                decoded_preds = self.bert_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = self.bert_tokenizer.batch_decode(labels, skip_special_tokens=True)

                # 打印数据观察一下结果
                # for pred_ele, label_ele in zip(decoded_preds, decoded_labels):
                #     print(pred_ele.encode("utf-8").decode("latin1"), "############", label_ele.encode("utf-8").decode("latin1"))

                decoded_preds = [pred.strip().replace(" ", "") for pred in decoded_preds]
                decoded_labels = [label.strip().replace(" ", "") for label in decoded_labels]
                
                all_pred_list.extend(decoded_preds)
                all_label_list.extend(decoded_labels)

        eval_loss = loss_total / len(data_loader)
        eval_cer_score = ASRScoreUtil.calculate_cer(all_pred_list, all_label_list)

        return eval_loss, eval_cer_score

    def test(self, data_loader):
        """
        模型测试
        @param:
        @return:
        """
        # 加载模型
        self.transformer_model.load_state_dict(torch.load(self.args.model_save_path, map_location=self.args.device))

        self.transformer_model.eval()
        self.transformer_model.to(self.args.device)
        if torch.cuda.device_count() > 1:
            self.transformer_model = torch.nn.DataParallel(self.transformer_model)

        _, eval_cer_score = self.evaluate(data_loader)
        LogUtil.logger.info("Test CER Score: {0}".format(eval_cer_score))

        # count = 0
        # with torch.no_grad():
        #     for step, batch_data in enumerate(data_loader):
        #         # 加入src_mask及tgt_mask输入
        #         self._add_mask(batch_data)                
        #         # 将数据加载到gpu(使用accelerator后可省略)
        #         batch_data = {k: v.to(self.args.device) for k, v in batch_data.items()}

        #         generated_tokens = self.batch_greedy_decode(
        #             batch_data["src"], batch_data["src_mask"], batch_data["src_padding_mask"],
        #             self.args.max_output_len, self.bert_tokenizer.cls_token_id, self.bert_tokenizer.sep_token_id)
                
        #         generated_tokens = np.array(generated_tokens)
        #         asrs = batch_data["src"].cpu().numpy()
        #         labels = batch_data["tgt"].cpu().numpy()

        #         # decode为字符
        #         decoded_preds = self.bert_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        #         decoded_asrs = self.bert_tokenizer.batch_decode(asrs, skip_special_tokens=True)
        #         decoded_labels = self.bert_tokenizer.batch_decode(labels, skip_special_tokens=True)

        #         for pred_ele, asr_ele, label_ele in zip(decoded_preds, decoded_asrs, decoded_labels):
        #             if pred_ele.strip().replace(" ", "") != asr_ele.strip().replace(" ", ""):
        #                 # print(pred_ele.encode("utf-8").decode("latin1"), "############", label_ele.encode("utf-8").decode("latin1"))
        #                 print(pred_ele.strip().replace(" ", "").encode("utf-8").decode("latin1"),
        #                  "############", asr_ele.strip().replace(" ", "").encode("utf-8").decode("latin1"), 
        #                  "############", label_ele.strip().replace(" ", "").encode("utf-8").decode("latin1"))
        #                 count += 1
                
        # print(count)

    def batch_greedy_decode(self, src, src_mask, src_padding_mask, max_len, start_symbol, end_symbol):
        """
        批量贪心解码
        @param:
        @return:
        """
        count = 0
        batch_size, src_seq_len = src.size()
        results = [[] for _ in range(batch_size)]
        stop_flag = [False for _ in range(batch_size)]
        
        # 将batch维度置于第二维
        src = src.transpose(0, 1)
        if torch.cuda.device_count() > 1:
            memory = self.transformer_model.module.encode(src, src_mask, src_padding_mask)
        else:
            memory = self.transformer_model.encode(src, src_mask, src_padding_mask)

        ys = torch.ones(batch_size, 1).fill_(start_symbol).type(torch.long).to(self.args.device)
        for _ in range(max_len-1):
            tgt_mask = self._generate_square_subsequent_mask(ys.size(1)).type(torch.bool).to(self.args.device)
            tgt_mask = tgt_mask.repeat(batch_size, 1, 1)
            # 将batch维度置于第二维
            ys = ys.transpose(0, 1)
            if torch.cuda.device_count() > 1:
                out = self.transformer_model.module.decode(ys, memory, tgt_mask, src_padding_mask)
            else:
                out = self.transformer_model.decode(ys, memory, tgt_mask, src_padding_mask)
            
            # 将batch置于第一维
            out = out.transpose(0, 1)
            if torch.cuda.device_count() > 1:
                prob = self.transformer_model.module.generator(out[:, -1, :])
            else:
                prob = self.transformer_model.generator(out[:, -1, :])
            pred = torch.argmax(prob, dim=-1)
            # 将batch维度置于第一维
            ys = ys.transpose(0, 1)
            ys = torch.cat((ys, pred.unsqueeze(1)), dim=1)
            pred = pred.cpu().numpy()
            for i in range(batch_size):
                if stop_flag[i] is False:
                    if pred[i] == end_symbol:
                        count += 1
                        stop_flag[i] = True
                    else:
                        results[i].append(pred[i].item())
                if count == batch_size:
                    break

        return results

    def greedy_decode(self, src, src_mask, max_len, start_symbol, end_symbol):
        """
        贪心解码, batch size为1
        @param:
        @return:
        """
        # 将batch维度置于第二维
        src = src.transpose(0, 1)
        
        if torch.cuda.device_count() > 1:
            memory = self.transformer_model.module.encode(src, src_mask, None)
        else:
            memory = self.transformer_model.encode(src, src_mask, None)
        
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(self.args.device)
        for i in range(max_len-1):
            # memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(self.args.device).type(torch.bool)
            tgt_mask = (self._generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(self.args.device)
            tgt_mask = tgt_mask.repeat(1, 1, 1)
            if torch.cuda.device_count() > 1:
                out = self.transformer_model.module.decode(ys, memory, tgt_mask, None)
            else:
                out = self.transformer_model.decode(ys, memory, tgt_mask, None)
                        
            # 将batch置于第一维，后续out[:, -1]则将中间长度维度消掉
            out = out.transpose(0, 1)
            if torch.cuda.device_count() > 1:
                prob = self.transformer_model.module.generator(out[:, -1])
            else:
                prob = self.transformer_model.generator(out[:, -1])
            
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)

            if next_word == end_symbol:
                break
        
        # 将batch维度置于第1维
        ys = ys.transpose(0, 1)
        return ys
