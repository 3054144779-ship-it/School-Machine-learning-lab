import os
import numpy as np
from knn import KNN
from Logistic import Logistic

def test1() :
    # 加载数据
    # 获取 main.py 所在的绝对目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(current_dir, 'KNN', 'data', 'datingTestSet2.txt')

    data = np.loadtxt(DATA_PATH, delimiter='\t')
    features = data[:, 0:3]
    labels = data[:, -1]

    # 归一化
    def auto_norm(dataSet):
        # 按行
        min_vals = dataSet.min(0)
        max_vals = dataSet.max(0)
        ranges = max_vals - min_vals
        norm_data = (dataSet - min_vals) / ranges
        return norm_data, ranges, min_vals

    norm_features, ranges, min_vals = auto_norm(features)

    # 划分数据集 
    m = norm_features.shape[0]
    num_test = int(m * 0.1) # 100条

    train_x = norm_features[num_test:m, :]
    train_y = labels[num_test:m]
    test_x = norm_features[0:num_test, :]
    test_y = labels[0:num_test]

    # 算法流程
    knn = KNN(k=20)  
    knn.fit(train_x, train_y)

    # 测试与评估
    error_rate = knn.auto_test(test_x, test_y)
    print(f"错误率为: {error_rate * 100:.2f}%")

    # 可视化 
    # ‘玩游戏时间’和‘吃冰淇淋数’作为坐标轴
    knn.plot_2d(norm_features, labels, x_idx=1, y_idx=2, title="Helen's Dating Data")

def test2(lr, iters) :
    current_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(current_dir, 'LOGISTIC', 'HorseColicTraining.txt')
    DATA_TEST_PATH = os.path.join(current_dir, 'LOGISTIC', 'HorseColicTest.txt')
    # 获取训练集
    data = np.loadtxt(DATA_PATH)

    features = data[:, 0:-1]
    labels = data[:, -1]

    log = Logistic(lr, iters) # 学习效率 和 迭代次数用默认值

    # 归一化
    # 算出特征的均值和标准差
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    # 边界处理(加不加都行，防止全零的，在这个test没用)
    std[std == 0] = 1e-8

    # 对特征做归一化
    features_scaled = (features - mean) / std

    log.fit(features_scaled, labels)

    testData = np.loadtxt(DATA_TEST_PATH)
    features_test = testData[:, 0:-1]
    labels_test = testData[:, -1]

    # 对测试集进行归一化
    features_test_scaled = (features_test - mean) / std
    # 进行预测 并且 预估错误率
    error_rate = log.auto_test(features_test_scaled, labels_test)
    # print(f"错误率为: {error_rate * 100:.2f}%")
    # log.plot_2d(features_test_scaled, labels_test, 3, 4, "Logistic")
    return error_rate

def test3():
    max_rate = 1.0
    best_lr = 0
    best_iters = 0

    # lr_list = np.arange(0.0001, 0.003, 0.0001)
    # iters_list = range(500, 1000, 100)
    lr_list = np.arange(0.01, 0.2, 0.02)
    iters_list = range(500, 8000, 200)

    total = len(lr_list) * len(iters_list)
    print(f"总共需要训练 {total} 次")

    for lr in lr_list:
        for iters in iters_list:
            rate = test2(lr, iters)

            if rate < max_rate:
                max_rate = rate
                best_lr = lr
                best_iters = iters

    print("="*50)
    print(f"最佳错误率 {max_rate:.4f}")
    print(f"最佳学习率：{best_lr}")
    print(f"最佳迭代次数：{best_iters}")

