#encoding: utf-8
'''
@File    :   test_data_process.py
@Time    :   2021/06/09 15:52:45
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import sys
sys.path.append("../../PhVEC")

from util.file_util import FileUtil
from util.asr_score_util import ASRScoreUtil

class TestDataProcess(object):
    """
    测试数据处理
    """

    def get_variable_data(self, data_path, variable_data_path):
        """
        获取变长数据集
        @param:
        @return:
        """
        data_obj_list = FileUtil.read_json_data(data_path)

        variable_obj_list = []
        asr_list = []
        transcript_list = []
        for data_obj in data_obj_list:
            if len(data_obj["asr"]) != len(data_obj["transcript"]):
                variable_obj_list.append(data_obj)
                asr_list.append(data_obj["asr"])
                transcript_list.append(data_obj["transcript"])
        
        print(ASRScoreUtil.calculate_cer(asr_list, transcript_list))
        FileUtil.write_json_data(variable_obj_list, variable_data_path)
    
    def generate_mlm_data(self, data_path, mlm_data_path):
        """
        生成mlm训练数据
        @param:
        @return:
        """
        data_obj_list = FileUtil.read_json_data(data_path)

        all_text_list = []
        len_list = []
        for data_obj in data_obj_list:
            # if len(data_obj["asr"]) != len(data_obj["transcript"]):
            #     continue
            
            text = data_obj["asr"] + "\t" + data_obj["transcript"]
            all_text_list.append(text)

            len_list.append(abs(len(data_obj["asr"])-len(data_obj["transcript"])))
        
        print(sum(len_list)/len(len_list))

        FileUtil.write_raw_data(all_text_list, mlm_data_path)

if __name__ == "__main__":
    test_data_path = "/ssd1/users/fangzheng/data/mt_error/correct_format/magic_test_filter.txt"
    variable_data_path = "/ssd1/users/fangzheng/data/mt_error/correct_format/magic_test_filter_variable.txt"

    test_data_process = TestDataProcess()
    # test_data_process.get_variable_data(test_data_path, variable_data_path)

    source_path = "/ssd1/users/fangzheng/data/mt_error/mlm_format/magic_test_filter_variable.txt"
    mlm_path = "/ssd1/users/fangzheng/data/mt_error/mlm_format/mlm_data/magic_test_filter_variable.txt"
    test_data_process.generate_mlm_data(source_path, mlm_path)




    

