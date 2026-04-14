import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# 负责画图
class MLVisualizerMixin :
    # 画出散点图
    def plot_2d(self, features, labels, x_idx=0, y_idx=0, title="text") :
        # 创建大小是 8*6 的
        plt.figure(figsize=(8, 6))
        # 数据位置 数据位置 颜色（根据标签分） 颜色映射表 点边缘颜色(k 是黑色) 不透明度
        plt.scatter(features[:, x_idx], features[:, y_idx], c=labels, cmap='viridis', edgecolors='k', alpha=0.7)
        # 名字
        plt.title(title)
        # 线条颜色
        plt.colorbar(label='Class')
        plt.show()

    # 画出二分图
    def plot_decision_boundary(self, features, labels, w, b):
        plt.figure(figsize=(8, 6))
        # 画点
        plt.scatter(features[:, 0], features[:, 1], c=labels.flatten(), cmap='viridis')

        # 画线: w1*x1 + w2*x2 + b = 0  =>  x2 = -(w1*x1 + b) / w2
        x1_values = np.linspace(np.min(features[:, 0]), np.max(features[:, 0]), 100)
        x2_values = -(w[0] * x1_values + b) / w[1]

        plt.plot(x1_values, x2_values, color='red', label='Decision Boundary')
        plt.legend()
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