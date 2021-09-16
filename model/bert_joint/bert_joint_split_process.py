#encoding: utf-8
'''
@File    :   bert_joint_split_process.py
@Time    :   2021/06/15 21:05:37
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

class BERTJointSplitProcess(object):
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
                detect_logits = self.bert_model((batch_data["detect_input_ids"], \
                    batch_data["detect_attention_mask"], batch_data["detect_type_ids"]), is_detect=True)
                correct_logits = self.bert_model((batch_data["correct_input_ids"], \
                    batch_data["correct_attention_mask"], batch_data["correct_type_ids"]), is_detect=False)

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
                char_list = [self.bert_tokenizer.convert_tokens_to_ids(char_ele) for char_ele in word+"".join(lazy_pinyin(word))]
                item_input_ids = item_input_ids[:index] + char_list + item_input_ids[index+1:]
                item_attention_mask = item_attention_mask[:index] + [1]*len(char_list) + item_attention_mask[index+1:]
                # char_list包含错误词
                offset += len(char_list) - 1
                offset_list.append(index)
            
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
                # 预测错误位置
                detect_logits = self.bert_model((batch_data["detect_input_ids"], \
                    batch_data["detect_attention_mask"], batch_data["detect_type_ids"]), is_detect=True)
                token_pred_ids = torch.argmax(detect_logits.detach(), dim=2)

                # 根据预测的错误位置构建拼音纠错模型输入
                batch_correct_input_tuple, all_error_offset_list = self.generate_correct_input(token_pred_ids.cpu().numpy().tolist(), \
                    batch_data["detect_input_ids"].cpu().numpy().tolist(), batch_data["detect_attention_mask"].cpu().numpy().tolist())
                
                # 纠正错误
                correct_logits = self.bert_model(batch_correct_input_tuple, is_detect=False)
                correct_pred_ids = torch.argmax(correct_logits.detach(), dim=2).cpu().numpy()
                transcript_labels = batch_data["transcript_labels"].cpu().numpy()

                # decode为字符
                decoded_preds = self.bert_tokenizer.batch_decode(correct_pred_ids, skip_special_tokens=True)
                decoded_labels = self.bert_tokenizer.batch_decode(transcript_labels, skip_special_tokens=True)

                # 将预测中的[unused1]标签及空格删除
                decoded_preds = [pred.strip().replace("[unused1]", "").replace(" ", "") for pred in decoded_preds]
                decoded_labels = [label.strip().replace(" ", "") for label in decoded_labels]

                all_pred_list.extend(decoded_preds)
                all_label_list.extend(decoded_labels)

                # 打印数据观察一下结果
                detect_input_sents = self.bert_tokenizer.batch_decode(batch_data["detect_input_ids"].cpu().numpy(), skip_special_tokens=True)
                detect_input_sents = [ele.strip().replace(" ", "") for ele in detect_input_sents]
                for input_ele, pred_ele, label_ele in zip(detect_input_sents, decoded_preds, decoded_labels):
                    if input_ele != pred_ele:
                        print(input_ele, "############", pred_ele, "############", label_ele)
                
                # # 打印attention结果
                # correct_inputs = self.bert_tokenizer.batch_decode(batch_correct_input_tuple[0].cpu().numpy())
                # for sent_attention, sent_input_id, offset_list in zip(attentions, correct_inputs, all_error_offset_list):
                #     if len(offset_list) > 0:
                #         for offset in offset_list:
                #             print(sent_input_id, "word", offset, sent_input_id.split()[offset])
                #             tmp_list = sent_attention[-1][offset].cpu().numpy().tolist()
                #             tmp_list0 = sent_attention[-1][offset-1].cpu().numpy().tolist()
                #             print(tmp_list, sum(tmp_list), tmp_list0, sum(tmp_list0))
                #             print("\n")

        # 计算CER
        cer_score = ASRScoreUtil.calculate_cer(all_pred_list, all_label_list)

        return cer_score
