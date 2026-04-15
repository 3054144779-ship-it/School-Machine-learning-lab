import numpy as np
import matplotlib as plt
from BaseAlgorithm import BaseAlgorithm

class Regression(BaseAlgorithm):
    def __init__(self, lr=0.01, iters=1000):
        super().__init__() 
        self.lr = lr
        self.iters = iters
        self.w = None
        self.b = None
    
    def _loss(self, z):
        # z.T dot z 得到平方和
        m = len(z)
        return np.dot(z.T, z) / (2 * m) 

    def fit(self, data, labels):
        # 行 列
        n, m = data.shape
        
        # 初始化权重 (m, 1) 和 偏置
        self.w = np.zeros((m, 1))
        self.b = 0
        y = labels.reshape(n, 1)

        for i in range(self.iters):
            # 预测值
            z = np.dot(data, self.w) + self.b
            
            # 误差
            error = z - y
            
            # 梯度
            dw = np.dot(data.T, error) / n
            db = np.sum(error) / n

            self.w -= self.lr * dw
            self.b -= self.lr * db

            # 是否收敛
            # if i % 100 == 0:
            #     loss = self.view_loss(error) 
                # print(f"Iter {i}: Loss {loss}")

    def predict(self, data):
        return np.dot(data, self.w) + self.b
    
    def auto_test(self, test_data, test_labels) :   
        preds = self.predict(test_data)
        # 统一成列向量
        y_true = test_labels.reshape(-1, 1)
        y_pred = preds.reshape(-1, 1)
        # 均方误差
        mse = np.mean((y_true - y_pred) ** 2)
        return mse
    
    """线性回归绘图：散点 + 拟合线"""
    def plot_regression_line(self, features, labels, x_idx=0):
        plt.figure(figsize=(8, 6))
        
        plt.scatter(features[:, x_idx], labels, color='blue', alpha=0.5, label='Actual Data')
        
        x_min, x_max = features[:, x_idx].min(), features[:, x_idx].max()
        x_line = np.linspace(x_min, x_max, 100).reshape(-1, 1)
        
        sorted_idx = features[:, x_idx].argsort()
        x_sorted = features[sorted_idx]
        y_pred = self.predict(x_sorted)
        
        plt.plot(x_sorted[:, x_idx], y_pred, color='red', linewidth=3, label='Regression Line')
        
        plt.title(f"Linear Regression Fit (Feature {x_idx})")
        plt.xlabel(f"Feature {x_idx}")
        plt.ylabel("Target Value")
        plt.legend()
        plt.show()