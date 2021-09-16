#encoding: utf-8
'''
@File    :   bert_single_ctc_process.py
@Time    :   2021/08/16 19:21:17
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import time

import torch
import numpy as np
from pypinyin import lazy_pinyin
from accelerate import Accelerator
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup

from util.log_util import LogUtil
from util.asr_score_util import ASRScoreUtil
from util.model_util import ModelUtil
from util.file_util import FileUtil
from util.text_util import TextUtil

class BERTCTCProcess(object):
    """
    BERT CTC 处理
    """
    """
    BERT模型训练、测试
    """
    def __init__(self, args, bert_model, bert_tokenizer):
        self.args = args
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.ctc_loss_fn = torch.nn.CTCLoss(blank=1, reduction="mean")

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

                correct_logits = self.bert_model((batch_data["input_ids"], \
                    batch_data["attention_mask"], batch_data["token_type_ids"]))
                
                correct_logits = correct_logits.transpose(0, 1).log_softmax(2).requires_grad_()
                loss = self.ctc_loss_fn(correct_logits, batch_data["labels"], batch_data["input_lengths"], batch_data["target_lengths"])

                loss.backward()
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
    
    def decode_ctc_output(self, batch_pred_list):
        """
        解码ctc字符串
        @param:
        @return:
        """
        clean_output_list = []
        for pred in batch_pred_list:
            pred_char_list = pred.strip().split(" ")
            pre_char = ""
            filter_pred_list = []
            for predict_char in pred_char_list:
                if predict_char != pre_char and predict_char != "[unused1]":
                    filter_pred_list.append(predict_char)
                pre_char = predict_char
            clean_output_list.append("".join(filter_pred_list).replace("[unused1]", ""))
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
                correct_logits = self.bert_model((batch_data["input_ids"], \
                    batch_data["attention_mask"], batch_data["token_type_ids"]))
                correct_pred_ids = torch.argmax(correct_logits.detach(), dim=2).cpu().numpy()
                transcript_labels = batch_data["labels"].cpu().numpy()

                # decode为字符
                decoded_preds = self.bert_tokenizer.batch_decode(correct_pred_ids, skip_special_tokens=True)
                decoded_labels = self.bert_tokenizer.batch_decode(transcript_labels, skip_special_tokens=True)

                # 将预测中的空格删除
                decoded_preds = self.decode_ctc_output(decoded_preds)
                decoded_labels = [label.strip().replace(" ", "") for label in decoded_labels]

                all_pred_list.extend(decoded_preds)
                all_label_list.extend(decoded_labels)

                # 打印数据观察一下结果
                # detect_input_sents = self.bert_tokenizer.batch_decode(batch_data["input_ids"].cpu().numpy(), skip_special_tokens=True)
                # detect_input_sents = [ele.strip().replace(" ", "") for ele in detect_input_sents]
                # all_asr_list.extend(detect_input_sents)
                # for input_ele, pred_ele, label_ele in zip(detect_input_sents, decoded_preds, decoded_labels):
                #     if input_ele != pred_ele:
                #         print(input_ele, "############", pred_ele, "############", label_ele)

        # 计算CER
        cer_score = ASRScoreUtil.calculate_cer(all_pred_list, all_label_list)

        # # 保存结果
        # save_sent_list = []
        # for asr_sent, pred_sent, label_sent in zip(all_asr_list, all_pred_list, all_label_list):
        #     if asr_sent != label_sent:
        #         save_sent_list.append("\t".join([asr_sent, pred_sent, label_sent]))
        # FileUtil.write_raw_data(save_sent_list, "result/single_asr_label_not_same.txt")

        return cer_score

    def evaluate2(self, data_loader):
        """
        验证模型, 检测时延
        @param:
        @return:
        """
        self.bert_model.eval()
        
        LogUtil.logger.info("######### Start Evaluate #########")
        start_time = time.time()

        with torch.no_grad():
            for step, batch_data in enumerate(data_loader):
                # 将数据加载到gpu(使用accelerator后可省略)
                batch_data = {k: v.to(self.args.device) for k, v in batch_data.items()}
                correct_logits = self.bert_model((batch_data["input_ids"], \
                    batch_data["attention_mask"], batch_data["token_type_ids"]))
                correct_pred_ids = torch.argmax(correct_logits.detach(), dim=2).cpu().numpy()

                # decode为字符
                decoded_preds = self.bert_tokenizer.batch_decode(correct_pred_ids, skip_special_tokens=True)
        
        LogUtil.logger.info("{0} ms".format((time.time()-start_time) * 1000))
        LogUtil.logger.info("######### End Evaluate #########")

        return 0

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
