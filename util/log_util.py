#encoding: utf-8
'''
@File    :   log_util.py
@Time    :   2021/04/20 16:48:43
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''

import logging

class LogUtil(object):
    """
    日志工具类
    """
    
    # 日志配置
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(filename)s line:%(lineno)d] %(levelname)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    logger = logging.getLogger()