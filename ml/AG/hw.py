import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score  
from Regression import Regression

def main():
    iris = datasets.load_iris()
    data = iris.data

    y = data[:, 0]
    X = data[:, 2].reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    X_train_mean = np.mean(X_train, axis=0)
    X_train_std = np.std(X_train, axis=0)
    X_train_scaled = (X_train - X_train_mean) / X_train_std
    X_test_scaled = (X_test - X_train_mean) / X_train_std

    model = Regression(lr=0.05, iters=500)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print("=== 模型评估结果 ===")
    print(f"拟合率 (R² Score): {r2:.4f}")
    print(f"均方误差 (MSE):    {mse:.4f}")
    print("====================")

    model.plot_regression_line(X_test_scaled, y_test, x_idx=0)

if __name__ == "__main__":
    main()