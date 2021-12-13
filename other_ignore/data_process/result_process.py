#encoding: utf-8
'''
@File    :   result_process.py
@Time    :   2021/06/24 11:18:07
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import sys
sys.path.append("../../PhVEC")


from util.file_util import FileUtil
from util.log_util import LogUtil

class ResultProcess(object):
    """
    结果处理
    """
    def analyse_two_result(self, other_result_path, our_result_path, compare_result_path):
        """
        结果对比分析
        @param:
        @return:
        """
        other_result_list = FileUtil.read_raw_data(other_result_path)
        our_result_list = FileUtil.read_raw_data(our_result_path)

        other_result_dict = {}
        for item in other_result_list:
            if len(item.split("\t")) != 3:
                continue
            asr, pred, label = item.split("\t")
            other_result_dict[asr] = pred
        
        diff_num = 0
        compare_result_list = []
        for item in our_result_list:
            if len(item.split("\t")) != 3:
                continue
            asr, pred, label = item.split("\t")
            other_pred = other_result_dict.get(asr, "")
            if other_pred != pred:
                diff_num += 1
                print(asr, pred, other_pred, label)
                compare_result_list.append("\t".join([asr, pred, other_pred, label]))
        
        LogUtil.logger.info("diff num: {0}".format(diff_num))
        FileUtil.write_raw_data(compare_result_list, compare_result_path)

if __name__ == "__main__":
    single_bert_result_path = "../result/single_asr_label_not_same.txt"
    joint_bert_result_path = "../result/joint_asr_label_not_same.txt"

    combine_path = "../result/joint_single_diff_result.txt"

    result_process = ResultProcess()
    result_process.analyse_two_result(single_bert_result_path, joint_bert_result_path, combine_path)
