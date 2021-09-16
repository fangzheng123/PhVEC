#encoding: utf-8
'''
@File    :   score_util.py
@Time    :   2021/05/17 20:11:55
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import editdistance
from itertools import groupby
import numpy as np
import six

class ASRScoreUtil(object):
    """
    Calculate CER and WER for ASR
    """
    @classmethod
    def calculate_cer(cls, seqs_hat, seqs_true):
        """Calculate sentence-level CER score.
        :param list seqs_hat: prediction
        :param list seqs_true: reference
        :return: average sentence-level CER score
        :rtype float
        """
        assert len(seqs_hat) == len(seqs_true)
        
        char_eds, char_ref_lens = [], []
        for seq_hat_text, seq_true_text in zip(seqs_hat, seqs_true):
            hyp_chars = seq_hat_text.replace(" ", "")
            ref_chars = seq_true_text.replace(" ", "")
            char_eds.append(editdistance.eval(hyp_chars, ref_chars))
            char_ref_lens.append(len(ref_chars))
        return float(sum(char_eds)) / sum(char_ref_lens)

    @classmethod
    def calculate_wer(cls, seqs_hat, seqs_true):
        """Calculate sentence-level WER score.
        :param list seqs_hat: prediction
        :param list seqs_true: reference
        :return: average sentence-level WER score
        :rtype float
        """
        assert len(seqs_hat) == len(seqs_true)

        word_eds, word_ref_lens = [], []
        for seq_hat_text, seq_true_text in zip(seqs_hat, seqs_true):
            hyp_words = " ".join(seq_hat_text.split())
            ref_words = " ".join(seq_true_text.split())
            word_eds.append(editdistance.eval(hyp_words, ref_words))
            word_ref_lens.append(len(ref_words))
        return float(sum(word_eds)) / sum(word_ref_lens)