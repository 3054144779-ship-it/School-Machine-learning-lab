import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier                                   # 分类决策树
from sklearn.linear_model import LinearRegression                                 # 多元线性回归
from sklearn.metrics import accuracy_score, confusion_matrix                      # 准确率 混淆矩阵
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error     # R² MAE RMSE

class Algorithm:
    def __init__(self):
        self.clf = DecisionTreeClassifier(max_depth=5, random_state=42)   # 初始化决策树深度设置为 5，随机数种子设置为 42 (防止过拟合)
        self.lr = LinearRegression()                                      # 初始化线性回归模型                                  
        self.feature_names = None                                         # 保存特征名称

    '''
    训练模型
    :param X_train: 训练集特征矩阵 (DataFrame)
    :param y_train_reg: 训练集目标变量 - 连续值 (用于线性回归，如具体分数)
    :param y_train_clf: 训练集目标变量 - 分类标签 (用于决策树，如优/良/中/差)
    '''
    def fit(self, X_train, y_train_reg, y_train_clf):
        # 如果传入的是 DataFrame，保存列名用于后续输出特征权重
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()

        # 分别用对应的数据训练两个模型
        self.lr.fit(X_train, y_train_reg)
        self.clf.fit(X_train, y_train_clf)

    '''
    预测数据
    :param X_test: 测试集特征矩阵
    :return: 两个模型的预测结果字典
    '''
    def predict(self, X_test):
        pred_reg = self.lr.predict(X_test)
        pred_clf = self.clf.predict(X_test)
        
        return {
            "regression_predictions": pred_reg,
            "classification_predictions": pred_clf
        }
    
    '''
    模型 A: 多元线性回归评估
    :param X_test: 测试集特征
    :param y_test_reg: 测试集真实连续分数
    '''
    def LinearRegression_assessment(self, X_test, y_test_reg):
        # 预测
        y_pred = self.lr.predict(X_test)
        
        # 计算评估指标
        r2 = r2_score(y_test_reg, y_pred)
        mae = mean_absolute_error(y_test_reg, y_pred)
        # RMSE 是 MSE 的平方根，所以嵌套一个 np.sqrt
        rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred)) 
        
        print("\n" + "="*30)
        print(" 模型 A (多元线性回归) 评估结果")
        print("="*30)
        print(f"R^2 (决定系数): {r2:.4f}")
        print(f"MAE (平均绝对误差): {mae:.4f}")
        print(f"RMSE (均方根误差): {rmse:.4f}")
        
        # 输出各特征的权重系数分析
        print("\n--- 特征权重 (Weights) 分析 ---")
        if self.feature_names:
            for name, weight in zip(self.feature_names, self.lr.coef_):
                direction = "正相关" if weight > 0 else "负相关"
                print(f"{name}: {weight:.4f} ({direction})")
        else:
            print(f"权重数组: {self.lr.coef_}")
            
        return r2, mae, rmse

    '''
    模型 B: 决策树评估
    :param X_test: 测试集特征
    :param y_test_clf: 测试集真实标签 (优/良/中/差)
    '''
    def tree_assessment(self, X_test, y_test_clf):
        # 预测
        y_pred = self.clf.predict(X_test)
        
        # 计算评估指标
        accuracy = accuracy_score(y_test_clf, y_pred)
        cm = confusion_matrix(y_test_clf, y_pred)
        
        print("\n" + "="*30)
        print(" 模型 B (分类决策树) 评估结果")
        print("="*30)
        print(f"准确率 (Accuracy): {accuracy * 100:.2f}%")
        print("混淆矩阵 (Confusion Matrix):")
        print(cm)
        
        return accuracy, cm
    