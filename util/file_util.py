#encoding: utf-8
'''
@File    :   file_util.py
@Time    :   2021/04/19 15:30:55
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import json

class FileUtil(object):
    """
    文件工具类
    """
    @classmethod
    def read_json_data(cls, data_path):
        """
        读取json数据
        @param:
        @return:
        """
        data_list = []
        with open(data_path, "r", encoding="utf-8") as data_file:
            for item in data_file:
                item = item.strip()
                data_json = json.loads(item)
                data_list.append(data_json)

        return data_list

    @classmethod
    def write_json_data(cls, data_list, data_path, is_indent=False):
        """
        写入json数据
        @param:
        @return:
        """
        # 是否格式化
        indent = None
        if is_indent:
            indent = 4
        
        with open(data_path, "w", encoding="utf-8") as data_file:
            for item in data_list:
                data_file.write(json.dumps(item, ensure_ascii=False, indent=indent) + "\n")

    @classmethod
    def read_raw_data(cls, data_path):
        """
        读取raw数据
        @param:
        @return:
        """
        data_list = []
        with open(data_path, "r", encoding="utf-8") as data_file:
            for item in data_file:
                item = item.strip()
                data_list.append(item)

                # if len(data_list) > 10000:
                #     break

        return data_list

    @classmethod
    def write_raw_data(cls, data_list, data_path):
        """
        写入raw数据
        @param:
        @return:
        """
        with open(data_path, "w", encoding="utf-8") as data_file:
            for item in data_list:
                data_file.write(item + "\n")
    

    @classmethod
    def write_json_in_append(cls, data_list, data_path):
        """
        接入形式写入json数据
        @param:
        @return:
        """
        with open(data_path, "a", encoding="utf-8") as data_file:
            for item in data_list:
                data_file.write(json.dumps(item, ensure_ascii=False) + "\n")