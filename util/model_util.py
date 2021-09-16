#encoding: utf-8
'''
@File    :   model_util.py
@Time    :   2021/06/08 10:26:26
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import torch
from collections import OrderedDict

class ModelUtil(object):
    """
    模型工具类
    """

    @classmethod
    def load_model(cls, model, model_save_path, device):
        """
        加载模型
        :param model: 模型对象
        :param model_save_path: 模型存储路径
        :param model_save_path: device
        :return:
        """
        # 当使用DataParallel训练时，key值会多出"module."
        state_dict = torch.load(model_save_path, map_location=device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # 移除 "module."
            if k.startswith("module."):
                k = k[7:]
            new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
    
    @classmethod
    def get_parameter_number(cls, model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
