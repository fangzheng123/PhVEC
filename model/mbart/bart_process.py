#encoding: utf-8
'''
@File    :   bart_process.py
@Time    :   2021/04/20 18:02:20
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import torch
import numpy as np

from datasets import load_metric
from accelerate import Accelerator
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup

from util.log_util import LogUtil

class BARTProcess(object):
    """
    BART模型训练、测试
    """
    def __init__(self, args, bart_model, bart_tokenizer):
        self.args = args
        self.bart_model = bart_model
        self.bart_tokenizer = bart_tokenizer
        self.bleu_metric = load_metric("sacrebleu")
        self.accelerator = Accelerator()

    def train(self, train_loader, dev_loader):
        """
        训练BART模型
        @param:
        @return:
        """
        self.bart_model.train()

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.bart_model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.bart_model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        t_total = len(train_loader) * self.args.num_train_epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * self.args.warmup_ratio), num_training_steps=t_total)

        # Prepare everything with "accelerator"
        self.bart_model, optimizer, train_loader, dev_loader = self.accelerator.prepare(
            self.bart_model, optimizer, train_loader, dev_loader
        )

        # 多GPU训练
        # if torch.cuda.device_count() > 1:
        #     self.bart_model = torch.nn.DataParallel(self.bart_model)
            
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
                # batch_data = {k: v.to(self.args.device) for k, v in batch_data.items()}
                seq2seq_output = self.bart_model(**batch_data)
                loss = seq2seq_output.loss
                self.accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # 输出在训练集和验证集上的效果
                LogUtil.logger.info(step)
                if total_batch % self.args.eval_batch_step == 0 and total_batch > 0:
                    dev_loss, dev_bleu_score = self.evaluate(dev_loader)
                    if dev_bleu_score > dev_best_score:
                        dev_best_score = dev_bleu_score
                        torch.save(self.accelerator.unwrap_model(self.bart_model).state_dict(), self.args.model_save_path)
                        improve = "*"
                        last_improve = total_batch
                    else:
                        improve = ""
                    msg = "Iter: {0:>3}, Train Loss: {1:>5.2}, Dev Loss: {2:>5.2},  Dev Bleu score: {3:>5.2} {4}"
                    LogUtil.logger.info(msg.format(total_batch, loss.item(), dev_loss.item(), dev_bleu_score, improve))
                    self.bart_model.train()

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
        self.bart_model.eval()
        
        loss_total = 0
        with torch.no_grad():
            for step, batch_data in enumerate(data_loader):
                # 将数据加载到gpu(使用accelerator后可省略)
                # batch_data = {k: v.to(self.args.device) for k, v in batch_data.items()}
                print(batch_data.shape)
                seq2seq_output = self.bart_model(**batch_data)
                loss = seq2seq_output.loss
                loss_total += loss
                
                generated_tokens = self.accelerator.unwrap_model(self.bart_model).generate(
                    input_ids=batch_data["input_ids"],
                    attention_mask=batch_data["attention_mask"],
                    max_length=self.args.max_output_len,
                    num_beams=self.args.num_beams
                )

                # padding输出
                generated_tokens = self.accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=self.bart_tokenizer.pad_token_id
                )
                # label已padding
                labels = batch_data["labels"]
                
                generated_tokens = self.accelerator.gather(generated_tokens).cpu().numpy()
                labels = self.accelerator.gather(labels).cpu().numpy()

                if self.args.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, self.bart_tokenizer.pad_token_id)

                # decode为字符
                decoded_preds = self.bart_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = self.bart_tokenizer.batch_decode(labels, skip_special_tokens=True)

                # 打印数据观察一下结果
                for pred_ele, label_ele in zip(decoded_preds, decoded_labels):
                    print(pred_ele.encode("utf-8").decode("latin1"), "############", label_ele.encode("utf-8").decode("latin1"))

                decoded_preds = [pred.strip() for pred in decoded_preds]
                decoded_labels = [[label.strip()] for label in decoded_labels]
                
                self.bleu_metric.add_batch(predictions=decoded_preds, references=decoded_labels)

        eval_loss = loss_total / len(data_loader)
        eval_bleu_score = self.bleu_metric.compute()["score"]

        return eval_loss, eval_bleu_score

    def test(self, test_loader):
        """
        测试模型
        @param:
        @return:
        """
        # 加载模型
        self.bart_model.load_state_dict(torch.load(self.args.model_save_path))

        self.bart_model.eval()
        test_loss, eval_bleu_score = self.evaluate(test_loader)
        LogUtil.logger.info("Test Loss: {0}, Test Bleu: {1}".format(test_loss, eval_bleu_score))




        





