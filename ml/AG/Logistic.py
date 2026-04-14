import numpy as np
from BaseAlgorithm import BaseAlgorithm

class Logistic(BaseAlgorithm) :
    def __init__(self, lr=0.01, iters=1000) :
        super().__init__()
        self.lr = lr # 学习率
        self.iters = iters # 迭代次数
        self.w = None # 权重
        self.b = None # 偏值

    # 激活函数
    def _sigmoid(self, z) :
        return 1 / (1 + np.exp(-z))

    # 训练
    def fit(self, data, labels) :
        # z = w^TX +b
        # 获取行列
        n, m = data.shape
        # 生成特征值是 0 的特征向量的转置矩阵（m 行 1 列）
        self.w = np.zeros((m, 1))
        self.b = 0
        # 统一标签维度为 (n, 1) （n 行 1 列）
        y = labels.reshape(n, 1)

        # 梯度下降迭代
        for i in range(self.iters) :
            # 线性预测值 (n 行 1 列)
            z = np.dot(data, self.w) + self.b
            # 概率(预测值)
            a = self._sigmoid(z)
            # 误差
            error = a - y
            # 梯度
            dw = (1 / n) * np.dot(data.T, error)
            db = (1 / n) * np.sum(error)

            self.w -= dw * self.lr
            self.b -= db * self.lr

    # 预测
    def predict(self, input_data) :
        x = np.array(input_data)
        z = np.dot(input_data, self.w) + self.b
        # 概率
        probability = self._sigmoid(z)
        # 阈值设置为 0.5
        return 1 if probability >= 0.5 else 0