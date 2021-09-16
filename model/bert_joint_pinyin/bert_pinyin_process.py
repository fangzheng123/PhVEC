#encoding: utf-8
'''
@File    :   bert_pinyin_process.py
@Time    :   2021/08/24 19:41:06
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import re
import time

import torch
import numpy as np
from pypinyin import lazy_pinyin
from accelerate import Accelerator
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup

from util.log_util import LogUtil
from util.model_util import ModelUtil
from util.file_util import FileUtil
from util.asr_score_util import ASRScoreUtil

class BERTPinyinProcess(object):
    """
    BERT模型训练、测试
    """
    def __init__(self, args, bert_model, bert_tokenizer):
        self.args = args
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.accelerator = Accelerator()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def train(self, train_loader, dev_loader):
        """
        训练BERT纠错模型
        @param:
        @return:
        """
        self.bert_model.train()

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.bert_model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.bert_model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        t_total = len(train_loader) * self.args.num_train_epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * self.args.warmup_ratio), num_training_steps=t_total)

        # Prepare everything with "accelerator", 公司服务器环境问题无法使用accelerator进行多卡训练
        # self.bert_model, optimizer, train_loader, dev_loader = self.accelerator.prepare(
        #     self.bert_model, optimizer, train_loader, dev_loader
        # )

        self.bert_model.to(self.args.device)
        # 多GPU训练
        if torch.cuda.device_count() > 1:
            self.bert_model = torch.nn.DataParallel(self.bert_model)
            
        # 进行到多少batch
        total_batch = 0
        dev_best_score = 100
        # 上次验证集loss下降的batch数
        last_improve = 0
        # 是否很久没有效果提升
        no_improve_flag = False

        LogUtil.logger.info("Batch Num: {0}".format(len(train_loader)))
        for epoch in range(self.args.num_train_epochs):
            LogUtil.logger.info("Epoch [{}/{}]".format(epoch + 1, self.args.num_train_epochs))
            for step, batch_data in enumerate(train_loader):
                # 将数据加载到gpu(使用accelerator后可省略)
                batch_data = {k: v.to(self.args.device) for k, v in batch_data.items()}
                detect_seq_output = self.bert_model((batch_data["detect_input_ids"], \
                    batch_data["detect_attention_mask"], batch_data["detect_type_ids"]))
                correct_seq_output = self.bert_model((batch_data["correct_input_ids"], \
                    batch_data["correct_attention_mask"], batch_data["correct_type_ids"]))

                if torch.cuda.device_count() > 1:
                    detect_logits = self.bert_model.module.error_detection(detect_seq_output)
                    correct_logits = self.bert_model.module.error_correction(correct_seq_output)
                else:
                    detect_logits = self.bert_model.error_detection(detect_seq_output)
                    correct_logits = self.bert_model.error_correction(correct_seq_output)
                
                detect_loss = self.loss_fn(detect_logits.reshape(-1, detect_logits.shape[-1]), batch_data["detect_labels"].reshape(-1))
                correct_loss = self.loss_fn(correct_logits.reshape(-1, correct_logits.shape[-1]), batch_data["correct_labels"].reshape(-1))
                loss = detect_loss + correct_loss
                loss.backward()
                # self.accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # 输出在训练集和验证集上的效果
                LogUtil.logger.info(step)
                if total_batch % self.args.eval_batch_step == 0 and total_batch > 0:
                    dev_cer_score = self.evaluate(dev_loader)
                    if dev_cer_score < dev_best_score:
                        dev_best_score = dev_cer_score
                        torch.save(self.bert_model.state_dict(), self.args.model_save_path)
                        # torch.save(self.accelerator.unwrap_model(self.bert_model).state_dict(), self.args.model_save_path)
                        improve = "*"
                        last_improve = total_batch
                    else:
                        improve = ""
                    msg = "Iter: {0:>3}, Train Loss: {1:>5.2},  Dev CER score: {2:>5.2} {3}"
                    LogUtil.logger.info(msg.format(total_batch, loss.item(), dev_cer_score, improve))
                    self.bert_model.train()

                total_batch += 1
                if total_batch - last_improve > self.args.require_improvement:
                    LogUtil.logger.info("No optimization for a long time, auto-stopping...")
                    no_improve_flag = True
                    break
            if no_improve_flag:
                break
    
    def generate_correct_input(self, detect_pred_ids, detect_input_ids, detect_attention_masks):
        """
        生成纠错输入
        @param:
        @return:
        """        
        all_input_id_list = []
        all_attention_mask_list = []
        all_type_id_list = []
        all_error_offset_list = []
        for item_pred_ids, item_input_ids, item_attention_mask in zip(detect_pred_ids, detect_input_ids, detect_attention_masks):
            # 抽取错误词
            error_index_word_list = [(index, self.bert_tokenizer.convert_ids_to_tokens(item_input_ids[index])) \
                for index, pred_id in enumerate(item_pred_ids) if pred_id == 0]

            # 提取错误词对应的拼音，并插入到对应位置
            offset = 0
            offset_list = []
            for index, word in error_index_word_list:
                index = index + offset
                char_list = [self.bert_tokenizer.convert_tokens_to_ids(char_ele) for char_ele in [word, lazy_pinyin(word)[0]]]
                item_input_ids = item_input_ids[:index] + char_list + item_input_ids[index+1:]
                item_attention_mask = item_attention_mask[:index] + [1]*len(char_list) + item_attention_mask[index+1:]
                # char_list包含错误词
                offset += len(char_list) - 1
                # 插入错误输入的起始位置
                offset_list.append((index, index+len(char_list)))
            
            all_error_offset_list.append(offset_list)
                
            # 打印观察一下构造的错误数据
            if len(error_index_word_list) > 0:
                correct_input_str = self.bert_tokenizer.decode(item_input_ids)
                print("错误输入:", correct_input_str, item_attention_mask, item_pred_ids)
                
            item_attention_mask = item_attention_mask[:self.args.max_input_len]
            item_input_ids = item_input_ids[:self.args.max_input_len]
            # 当[SEP]被截取掉时，将最后一位转化为[SEP]
            if item_input_ids[-1] not in [0, 102]:
                item_input_ids[-1] = 102

            all_input_id_list.append(item_input_ids)
            all_attention_mask_list.append(item_attention_mask)
            all_type_id_list.append([0] * self.args.max_input_len)
            # correct_input_list.append(input_str)
        
        # batch_correct_input = self.bert_tokenizer(correct_input_list, truncation=True, padding="max_length", max_length=self.args.max_detect_input_len)
        
        # 转化为tensor
        input_ids = torch.LongTensor(all_input_id_list)
        attention_mask = torch.LongTensor(all_attention_mask_list)
        token_type_ids = torch.LongTensor(all_type_id_list)
        batch_correct_input_tuple = (input_ids, attention_mask, token_type_ids)
        batch_correct_input_tuple = tuple([ele.to(self.args.device) for ele in batch_correct_input_tuple])

        return batch_correct_input_tuple, all_error_offset_list
    
    def decode_hand_output(self, correct_pred_ids, all_error_offset_list):
        """
        解码ctc字符串
        @param:
        @return:
        """
        clean_output_list = []
        for pred_ids, error_offset_list in zip(correct_pred_ids, all_error_offset_list):
            # 正确句子解码
            if len(error_offset_list) == 0:
                pred_text = self.bert_tokenizer.decode(pred_ids, skip_special_tokens=True)
            # 错误句子需要对重复字去重
            else:
                previous_end = 0
                filter_id_list = []
                for (start, end) in error_offset_list:
                    filter_id_list.extend(pred_ids[previous_end:start])
                    pred_id_set = set()
                    for pred_id in pred_ids[start:end]:
                        # 不重复则保留(1个错字不允许出现2个重复label)
                        if pred_id not in pred_id_set:
                            filter_id_list.append(pred_id)
                        pred_id_set.add(pred_id)
                    previous_end = end
                filter_id_list.extend(pred_ids[previous_end:])
                pred_text = self.bert_tokenizer.decode(filter_id_list, skip_special_tokens=True)
            
            clean_output_list.append(pred_text.strip().replace(" ", ""))
                    
        return clean_output_list

    def evaluate(self, data_loader):
        """
        验证模型
        @param:
        @return:
        """
        self.bert_model.eval()
        
        cer_score = 0.0
        all_pred_list = []
        all_label_list = []
        all_asr_list = []
        with torch.no_grad():
            for step, batch_data in enumerate(data_loader):
                # 将数据加载到gpu(使用accelerator后可省略)
                batch_data = {k: v.to(self.args.device) for k, v in batch_data.items()}
                # 预测错误位置
                detect_seq_output = self.bert_model((batch_data["detect_input_ids"], \
                    batch_data["detect_attention_mask"], batch_data["detect_type_ids"]))
                
                if torch.cuda.device_count() > 1:
                    detect_logits = self.bert_model.module.error_detection(detect_seq_output)
                else:
                    detect_logits = self.bert_model.error_detection(detect_seq_output)
                token_pred_ids = torch.argmax(detect_logits.detach(), dim=2)

                # 根据预测的错误位置构建拼音纠错模型输入
                batch_correct_input_tuple, all_error_offset_list = self.generate_correct_input(token_pred_ids.cpu().numpy().tolist(), \
                    batch_data["detect_input_ids"].cpu().numpy().tolist(), batch_data["detect_attention_mask"].cpu().numpy().tolist())
                
                # 纠正错误
                correct_seq_output = self.bert_model(batch_correct_input_tuple)
                if torch.cuda.device_count() > 1:
                    correct_logits = self.bert_model.module.error_correction(correct_seq_output)
                else:
                    correct_logits = self.bert_model.error_correction(correct_seq_output)
                correct_pred_ids = torch.argmax(correct_logits.detach(), dim=2).cpu().numpy()
                transcript_labels = batch_data["transcript_labels"].cpu().numpy()

                # 解码预测结果
                decoded_preds = self.decode_hand_output(correct_pred_ids, all_error_offset_list)
                # decoded_preds = self.bert_tokenizer.batch_decode(correct_pred_ids, skip_special_tokens=True)
                decoded_preds = [pred.strip().replace("[UNK]", "").replace(" ", "") for pred in decoded_preds]

                # 解码标签结果
                decoded_labels = self.bert_tokenizer.batch_decode(transcript_labels, skip_special_tokens=True)
                decoded_labels = [label.strip().replace(" ", "") for label in decoded_labels]

                all_pred_list.extend(decoded_preds)
                all_label_list.extend(decoded_labels)

                # 打印数据观察一下结果
                detect_input_sents = self.bert_tokenizer.batch_decode(batch_data["detect_input_ids"].cpu().numpy(), skip_special_tokens=True)
                detect_input_sents = [ele.strip().replace(" ", "") for ele in detect_input_sents]
                all_asr_list.extend(detect_input_sents)
                for input_ele, pred_ele, label_ele in zip(detect_input_sents, decoded_preds, decoded_labels):
                    if input_ele != pred_ele:
                        print(input_ele, "############", pred_ele, "############", label_ele)
                
                # 打印attention
                # self.output_attention_val(attentions, batch_correct_input_tuple[0], all_error_offset_list)
                
                # 保存token embedding
                # self.output_token_embed(correct_seq_output, batch_correct_input_tuple[0], all_error_offset_list)
            
        # 计算CER
        cer_score = ASRScoreUtil.calculate_cer(all_pred_list, all_label_list)

        # 保存打印结果
        # if not self.args.do_train
        #     self.save_sent_result(all_asr_list, all_pred_list, all_label_list)

        return cer_score
    
    def test(self, test_loader):
        """
        测试模型
        @param:
        @return:
        """
        # 加载模型
        ModelUtil.load_model(self.bert_model, self.args.model_save_path, self.args.device)

        self.bert_model.eval()
        self.bert_model.to(self.args.device)
        if torch.cuda.device_count() > 1:
            self.bert_model = torch.nn.DataParallel(self.bert_model)

        eval_cer_score = self.evaluate(test_loader)
        LogUtil.logger.info("Test CER Score: {0}".format(eval_cer_score))

        # self.output_sent_attention()
        # self.output_bert_token_embedding()

    

                    
                    