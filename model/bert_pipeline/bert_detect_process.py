#encoding: utf-8
'''
@File    :   bert_detect_process.py
@Time    :   2021/06/09 16:38:16
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import torch
import numpy as np
from pypinyin import lazy_pinyin
from accelerate import Accelerator
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup

from util.log_util import LogUtil
from util.model_util import ModelUtil
from util.asr_score_util import ASRScoreUtil

class BERTDetectProcess(object):
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
        训练BERT探测模型
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
        dev_best_score = 0
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
                
                detect_logits = self.bert_model((batch_data["input_ids"], \
                    batch_data["attention_mask"], batch_data["token_type_ids"]))
                
                loss = self.loss_fn(detect_logits.reshape(-1, detect_logits.shape[-1]), batch_data["labels"].reshape(-1))
                loss.backward()
                # self.accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # 输出在训练集和验证集上的效果
                LogUtil.logger.info(step)
                if total_batch % self.args.eval_batch_step == 0 and total_batch > 0:
                    result_dict = self.evaluate(dev_loader)
                    dev_f1_score = result_dict["f1"]
                    if dev_best_score < dev_f1_score:
                        dev_best_score = dev_f1_score
                        torch.save(self.bert_model.state_dict(), self.args.model_save_path)
                        # torch.save(self.accelerator.unwrap_model(self.bert_model).state_dict(), self.args.model_save_path)
                        improve = "*"
                        last_improve = total_batch
                    else:
                        improve = ""
                    msg = "Iter: {0:>3}, Train Loss: {1:>5.2},  Dev Precision score: {2:>5.2}, Dev Recall score: {3:>5.2}, Dev F1 score: {4:>5.2}, {5}"
                    LogUtil.logger.info(msg.format(total_batch, loss.item(), result_dict["precision"], result_dict["recall"], result_dict["f1"], improve))
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
        
        all_pred_list = []
        all_label_list = []
        with torch.no_grad():
            for step, batch_data in enumerate(data_loader):
                # 将数据加载到gpu(使用accelerator后可省略)
                batch_data = {k: v.to(self.args.device) for k, v in batch_data.items()}
                
                # 预测错误位置
                detect_logits = self.bert_model((batch_data["input_ids"], \
                    batch_data["attention_mask"], batch_data["token_type_ids"]))
                token_pred_ids = torch.argmax(detect_logits.detach(), dim=2).cpu().numpy().tolist()
                
                token_labels = batch_data["labels"].cpu().numpy().tolist()
                
                # 计算P,R,F1
                for pred_list, label_list in zip(token_pred_ids, token_labels):
                    # 转化成NER任务使用的BIOS格式
                    all_pred_list.append(["O" if ele==1 else "S-Dec" for ele in pred_list])
                    all_label_list.append(["O" if ele==1 else "S-Dec" for ele in label_list])

                # decode为字符
                decoded_inputs = self.bert_tokenizer.batch_decode(batch_data["input_ids"].cpu().numpy(), skip_special_tokens=True)
                decoded_transcripts = self.bert_tokenizer.batch_decode(batch_data["transcript"].cpu().numpy(), skip_special_tokens=True)
                decoded_inputs = [input.strip().replace(" ", "") for input in decoded_inputs]
                decoded_transcripts = [transcript.strip().replace(" ", "") for transcript in decoded_transcripts]

                # 打印数据观察一下结果
                for input_ele, pred_ele, label_ele in zip(decoded_inputs, token_pred_ids, decoded_transcripts):
                    if input_ele != label_ele:
                        print(input_ele, "############", pred_ele, "############", label_ele)

        # 计算P, R, F
        result_dict = {
            "precision": precision_score(all_label_list, all_pred_list),
            "recall": recall_score(all_label_list, all_pred_list),
            "f1": f1_score(all_label_list, all_pred_list)
        }

        return result_dict

    def predict(self, test_loader):
        """
        预测模型
        @param:
        @return:
        """
        # 加载模型
        ModelUtil.load_model(self.bert_model, self.args.detect_model_path, self.args.device)

        self.bert_model.eval()
        self.bert_model.to(self.args.device)
        if torch.cuda.device_count() > 1:
            self.bert_model = torch.nn.DataParallel(self.bert_model)

        all_token_pred_list = []
        all_input_id_list = []
        all_attention_mask_list = []
        all_transcript_list = []
        with torch.no_grad():
            for step, batch_data in enumerate(test_loader):
                # 将数据加载到gpu(使用accelerator后可省略)
                batch_data = {k: v.to(self.args.device) for k, v in batch_data.items()}
                
                # 预测错误位置
                detect_logits = self.bert_model((batch_data["input_ids"], \
                    batch_data["attention_mask"], batch_data["token_type_ids"]))
                token_pred_ids = torch.argmax(detect_logits.detach(), dim=2).cpu().numpy().tolist()

                all_token_pred_list.extend(token_pred_ids)
                all_input_id_list.extend(batch_data["input_ids"].cpu().numpy().tolist())
                all_attention_mask_list.extend(batch_data["attention_mask"].cpu().numpy().tolist())
                all_transcript_list.extend(batch_data["transcript"].cpu().numpy().tolist())
        
        return self.generate_correct_input(all_token_pred_list, all_input_id_list, all_attention_mask_list, all_transcript_list)
                
    def generate_correct_input(self, detect_pred_ids, detect_input_ids, detect_attention_masks, detect_transcript_list):
        """
        生成纠错输入
        @param:
        @return:
        """        
        all_detect_input_list = []
        all_correct_input_list = []
        all_attention_mask_list = []
        all_type_id_list = []
        all_error_index_list = []
        all_transcript_list = []

        right_input_list = []
        for item_pred_ids, item_input_ids, item_attention_mask, transcript_ids in zip(detect_pred_ids, detect_input_ids, detect_attention_masks, detect_transcript_list):
            # 抽取错误词
            error_index_word_list = [(index, self.bert_tokenizer.convert_ids_to_tokens(item_input_ids[index])) \
                for index, pred_id in enumerate(item_pred_ids) if pred_id == 0]
            
            if len(error_index_word_list) == 0:
                right_input_list.append((item_input_ids, transcript_ids))
            else:
                all_detect_input_list.append(item_input_ids[:])
                
                # 提取错误词对应的拼音，并插入到对应位置
                offset = 0
                for index, word in error_index_word_list:
                    index = index + offset
                    char_list = [self.bert_tokenizer.convert_tokens_to_ids(char_ele) for char_ele in word+"".join(lazy_pinyin(word))]
                    item_input_ids = item_input_ids[:index] + char_list + item_input_ids[index+1:]
                    item_attention_mask = item_attention_mask[:index] + [1]*len(char_list) + item_attention_mask[index+1:]
                    # char_list包含错误词
                    offset += len(char_list) - 1
                    
                # 打印观察一下构造的错误数据
                correct_input_str = self.bert_tokenizer.decode(item_input_ids)
                print("错误输入:", correct_input_str, self.bert_tokenizer.decode(transcript_ids, skip_special_tokens=True).strip().replace(" ", ""))
                    
                item_attention_mask = item_attention_mask[:self.args.max_input_len]
                item_input_ids = item_input_ids[:self.args.max_input_len]
                # 当[SEP]被截取掉时，将最后一位转化为[SEP]
                if item_input_ids[-1] not in [0, 102]:
                    item_input_ids[-1] = 102

                all_correct_input_list.append(item_input_ids)
                all_attention_mask_list.append(item_attention_mask)
                all_type_id_list.append([0] * self.args.max_input_len)
                # all_error_index_list.append([index for index, _ in error_index_word_list])
                all_transcript_list.append(transcript_ids)
                
        # 转化为tensor
        correct_input_ids = torch.LongTensor(all_correct_input_list)
        attention_mask = torch.LongTensor(all_attention_mask_list)
        token_type_ids = torch.LongTensor(all_type_id_list)
        all_correct_input_tuple = (correct_input_ids, attention_mask, token_type_ids, all_transcript_list)

        LogUtil.logger.info("错误句子数量: {0}".format(len(all_correct_input_list)))

        return right_input_list, all_correct_input_tuple