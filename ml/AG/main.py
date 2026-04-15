import os
import numpy as np
from knn import KNN
from Logistic import Logistic
from Regression import Regression

def test1():
    # 获取 main.py 所在的绝对目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(current_dir, 'KNN', 'data', 'datingTestSet2.txt')

    data = np.loadtxt(DATA_PATH, delimiter='\t')
    features = data[:, 0:3]
    labels = data[:, -1]

    # 归一化
    def auto_norm(dataSet):
        min_vals = dataSet.min(0)
        max_vals = dataSet.max(0)
        ranges = max_vals - min_vals
        norm_data = (dataSet - min_vals) / ranges
        return norm_data, ranges, min_vals

    norm_features, ranges, min_vals = auto_norm(features)

    # 划分数据集 
    m = norm_features.shape[0]
    num_test = int(m * 0.1) 

    train_x = norm_features[num_test:m, :]
    train_y = labels[num_test:m]
    test_x = norm_features[0:num_test, :]
    test_y = labels[0:num_test]

    # 训练
    knn = KNN(k=20)  
    knn.fit(train_x, train_y)

    # 测试与评估
    error_rate = knn.auto_test(test_x, test_y)
    print(f"KNN 错误率为: {error_rate * 100:.2f}%")

    # 可视化 
    knn.plot_2d(norm_features, labels, x_idx=1, y_idx=2, title="Helen's Dating Data")

def get_min_error_rate(func):
    def wrapper():
        max_rate = float('inf')
        best_lr = 0
        best_iters = 0

        lr_list = np.arange(0.05, 0.3, 0.02)
        iters_list = range(500, 20000, 2000) 

        total = len(lr_list) * len(iters_list)
        print(f"总共需要训练 {total} 次...")

        for lr in lr_list:
            for iters in iters_list:
                rate = func(lr, iters)
                if rate < max_rate:
                    max_rate = rate
                    best_lr = lr
                    best_iters = iters

        print("="*50)
        print(f"最佳评估值 (MSE/Error): {max_rate:.4f}")
        print(f"最佳学习率：{best_lr}")
        print(f"最佳迭代次数：{best_iters}")
    return wrapper

@get_min_error_rate
def test2(lr=0.01, iters=500):
    # 存入缓存，防止频繁 IO
    if not hasattr(test2, "cache"):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        DATA_PATH = os.path.join(current_dir, 'LOGISTIC', 'HorseColicTraining.txt')
        DATA_TEST_PATH = os.path.join(current_dir, 'LOGISTIC', 'HorseColicTest.txt')
        
        data = np.loadtxt(DATA_PATH)
        features = data[:, 0:-1]
        labels = data[:, -1]

        # 归一化
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std[std == 0] = 1e-8

        # 预存到缓存
        testData = np.loadtxt(DATA_TEST_PATH)
        test2.cache = {
            "train_x": (features - mean) / std,
            "train_y": labels,
            "test_x": (testData[:, 0:-1] - mean) / std,
            "test_y": testData[:, -1]
        }

    d = test2.cache
    log = Logistic(lr, iters)
    log.fit(d["train_x"], d["train_y"])
    return log.auto_test(d["test_x"], d["test_y"])

@get_min_error_rate
def test3(lr=0.29, iters=16500):
    if not hasattr(test3, "cache"):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        DATA_PATH = os.path.join(current_dir, 'REGRESSION', 'abalone.txt')

        data = np.loadtxt(DATA_PATH)
        features = data[:, 0:-1]
        labels = data[:, -1]
        
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std[std == 0] = 1e-8
        features_scaled = (features - mean) / std

        m = features_scaled.shape[0]
        num_test = int(m * 0.1) 
        
        test3.cache = {
            "train_x": features_scaled[num_test:],
            "train_y": labels[num_test:],
            "test_x": features_scaled[:num_test],
            "test_y": labels[:num_test]
        }

    d = test3.cache
    reg = Regression(lr, iters)
    reg.fit(d["train_x"], d["train_y"])
    return reg.auto_test(d["test_x"], d["test_y"])


# if __name__ == "__main__":
    # test1() 
    # test2()
    # test3()   