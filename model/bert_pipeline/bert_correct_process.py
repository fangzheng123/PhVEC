#encoding: utf-8
'''
@File    :   bert_process.py
@Time    :   2021/05/31 17:01:17
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import torch
import numpy as np
from pypinyin import lazy_pinyin
from accelerate import Accelerator
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup

from util.log_util import LogUtil
from util.model_util import ModelUtil
from util.asr_score_util import ASRScoreUtil

class BERTCorrectProcess(object):
    """
    BERT模型训练、测试
    """
    def __init__(self, args, bert_model, bert_tokenizer):
        self.args = args
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
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
                correct_logits = self.bert_model((batch_data["input_ids"], \
                    batch_data["attention_mask"], batch_data["token_type_ids"]))
                
                loss = self.loss_fn(correct_logits.reshape(-1, correct_logits.shape[-1]), batch_data["labels"].reshape(-1))
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
        with torch.no_grad():
            for step, batch_data in enumerate(data_loader):
                # 将数据加载到gpu(使用accelerator后可省略)
                batch_data = {k: v.to(self.args.device) for k, v in batch_data.items()}
                
                # 纠正错误
                correct_logits = self.bert_model((batch_data["input_ids"], \
                    batch_data["attention_mask"], batch_data["token_type_ids"]))
                
                correct_pred_ids = torch.argmax(correct_logits.detach(), dim=2).cpu().numpy()
                correct_labels = batch_data["labels"].cpu().numpy()
                transcript_labels = batch_data["transcript"].cpu().numpy()

                # decode为字符
                decoded_preds = self.bert_tokenizer.batch_decode(correct_pred_ids, skip_special_tokens=True)
                decoded_labels = self.bert_tokenizer.batch_decode(correct_labels, skip_special_tokens=True)
                decoded_transcripts = self.bert_tokenizer.batch_decode(transcript_labels, skip_special_tokens=True)

                # 将预测中的[unused1]标签及空格删除
                decoded_preds = [pred.strip().replace("[unused1]", "").replace(" ", "") for pred in decoded_preds]
                decoded_labels = [label.strip().replace("[unused1]", "").replace(" ", "") for label in decoded_labels]
                decoded_transcripts = [label.strip().replace(" ", "") for label in decoded_transcripts]

                all_pred_list.extend(decoded_preds)
                all_label_list.extend(decoded_labels)

                # 打印数据观察一下结果
                for pred_ele, label_ele, transcript_ele in zip(decoded_preds, decoded_labels, decoded_transcripts):
                    if pred_ele != label_ele:
                        print(pred_ele, "############", label_ele, "############", transcript_ele)

        # 计算CER
        cer_score = ASRScoreUtil.calculate_cer(all_pred_list, all_label_list)

        return cer_score

    def predict(self, right_input_list, all_correct_input_tuple):
        """
        预测模型
        @param:
        @return:
        """
        # 加载模型
        ModelUtil.load_model(self.bert_model, self.args.correct_model_path, self.args.device)

        self.bert_model.eval()
        self.bert_model.to(self.args.device)
        if torch.cuda.device_count() > 1:
            self.bert_model = torch.nn.DataParallel(self.bert_model)

        all_pred_list = []
        all_label_list = []

        correct_input_ids, all_attention_mask, all_token_type_ids, all_transcript_list = all_correct_input_tuple
        batch_num = len(all_transcript_list) // self.args.eval_batch_size + 1
        with torch.no_grad():
            for index in range(batch_num):
                begin_index = index*self.args.eval_batch_size
                end_index = (index+1)*self.args.eval_batch_size
                batch_data = (correct_input_ids[begin_index:end_index], all_attention_mask[begin_index:end_index], all_token_type_ids[begin_index:end_index])
                batch_data = tuple([ele.to(self.args.device) for ele in batch_data])

                # 纠正错误
                correct_logits = self.bert_model(batch_data)
                correct_pred_ids = torch.argmax(correct_logits.detach(), dim=2).cpu().numpy().tolist()
                
                transcript_labels = all_transcript_list[begin_index:end_index]

                # decode为字符
                decoded_preds = self.bert_tokenizer.batch_decode(correct_pred_ids, skip_special_tokens=True)
                decoded_labels = self.bert_tokenizer.batch_decode(transcript_labels, skip_special_tokens=True)
                # 将预测中的[unused1]标签及空格删除
                decoded_preds = [pred.strip().replace("[unused1]", "").replace(" ", "") for pred in decoded_preds]
                decoded_labels = [label.strip().replace(" ", "") for label in decoded_labels]
                
                all_pred_list.extend(decoded_preds)
                all_label_list.extend(decoded_labels)

                # 打印数据观察一下结果
                for pred_ele, label_ele in zip(decoded_preds, decoded_labels):
                    print(pred_ele, "############", label_ele)

        LogUtil.logger.info("预测句子数量: {0}".format(len(all_pred_list)))

        # 加入检测器预测为完全正确的句子
        for rigth_input_id, transcript_id in right_input_list:
            decoded_pred = self.bert_tokenizer.decode(rigth_input_id, skip_special_tokens=True)
            decoded_label = self.bert_tokenizer.decode(transcript_id, skip_special_tokens=True)
            all_pred_list.append(decoded_pred.strip().replace(" ", ""))
            all_label_list.append(decoded_label.strip().replace(" ", ""))

        LogUtil.logger.info("预测句子数量: {0}".format(len(all_pred_list)))

        # 计算CER
        cer_score = ASRScoreUtil.calculate_cer(all_pred_list, all_label_list)

        return cer_score