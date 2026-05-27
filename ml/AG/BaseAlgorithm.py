import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# 负责画图
class MLVisualizerMixin :
    def plot_2d(self, features, labels, x_idx=0, y_idx=0, title="text") :
        # 创建大小是 8*6 的
        plt.figure(figsize=(8, 6))
        # 数据位置 数据位置 大小 颜色 注册图颜色名称 边缘颜色 不透明度
        plt.scatter(features[:, x_idx], features[:, y_idx], c=labels, cmap='viridis', edgecolors='k', alpha=0.7)
        # 标签
        plt.title(title)
        # 线条颜色
        plt.colorbar(label='Class')
        plt.show()

class BaseAlgorithm(ABC, MLVisualizerMixin):
    def __init__(self):
        # 生成数据
        self.train_data = None
        # 数据标签
        self.train_labels = None

    """训练过程"""
    @abstractmethod
    def fit(self, data, labels):
        pass

    """预测过程"""
    @abstractmethod
    def predict(self, input_data):
        pass

    # 生成数据 数据标签 
    def auto_test(self, test_data, test_labels): 
        """通用测试逻辑：计算错误率"""
        error_count = 0
        for i in range(len(test_data)):
            res = self.predict(test_data[i])
            if res != test_labels[i]:
                error_count += 1
        return error_count / len(test_data)