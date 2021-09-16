#encoding: utf-8
'''
@File    :   tsne.py
@Time    :   2021/06/16 16:44:23
@Author  :   fangzheng.eric
@Contact :   fangzheng01@baidu.com
'''
import sys
sys.path.append("../../MTError")

from time import time

import numpy as np
from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from util.file_util import FileUtil
from util.log_util import LogUtil

class TSNEFigure(object):
    """
    绘制T-SNE图
    """

    def load_data(self, data_path):
        """
        加载token向量数据
        @param:
        @return:
        """
        data_list = FileUtil.read_json_data(data_path)
        fea_list = []
        label_list = []

        for ele_list in data_list:
            fea_list.append(ele_list[-1])
            label_list.append(ele_list[1])

        return np.array(fea_list), np.array(label_list)

    def plot_embedding(self, data, label, title):
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)

        plt.figure()
        ax = plt.subplot(111)
        colors = ["r", "b"]
        for i in range(data.shape[0]):
            plt.scatter(data[i, 0], data[i, 1], c=colors[label[i]], alpha=0.9)
            # plt.text(data[i, 0], data[i, 1], str(label[i]),
            #          color=plt.cm.Set1(label[i]),
            #          fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.title(title)
        plt.show()

    def main(self, data_path):
        fea_array, label_array = self.load_data(data_path)

        LogUtil.logger.info("Computing t-SNE embedding")
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        result_array = tsne.fit_transform(fea_array)

        LogUtil.logger.info("开始绘制")
        self.plot_embedding(result_array, label_array, "t-SNE embedding of the word")
        LogUtil.logger.info("绘制完毕")

if __name__ == "__main__":
    t_sne = TSNEFigure()
    t_sne.main("/ssd1/users/fangzheng/data/mt_error/correct_format/token_embed.txt")
