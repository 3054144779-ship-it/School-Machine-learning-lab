import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder 
from knn import KNN
from Logistic import Logistic
from Regression import Regression
from Decision_Tree import DecisionTree
from sklearn.model_selection import train_test_split 

current_dir = os.path.dirname(os.path.abspath(__file__))

def test1():
    # 获取 main.py 所在的绝对目录
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

# ==========================================
# 决策树专属：通用数据清洗函数
# ==========================================
def load_and_clean_data(file_path):
    """
    通用数据加载函数：使用 pandas 读取文件，并将所有字符串特征转换为数字
    自动检测分隔符：Tab 分隔则用 \\t，否则用连续空白 \\s+
    """
    with open(file_path, 'r') as f:
        first_line = f.readline()
    sep = r'\t' if '\t' in first_line else r'\s+'

    df = pd.read_csv(file_path, header=None, sep=sep, engine='python')
    df = df.dropna(axis=1, how='all')

    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])

    return df.values


def run_tree_test(name, file_name, max_depth=5, test_size=0.2, rs=1):
    """通用决策树测试：加载 -> 清洗 -> 划分 -> 训练 -> 评估 -> 可视化"""
    DATA_PATH = os.path.join(current_dir, 'DecisionTree', file_name)
    print(f"\n{'='*50}")
    print(f"数据集: {name} ({file_name})")
    print(f"{'='*50}")
    print("正在加载并清洗数据...")
    data = load_and_clean_data(DATA_PATH)
    n_samples, n_features = data.shape[0], data.shape[1] - 1
    n_classes = len(np.unique(data[:, -1]))
    print(f"样本数: {n_samples}, 特征数: {n_features}, 类别数: {n_classes}")

    features = data[:, 0:-1]
    labels = data[:, -1]

    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=test_size, random_state=rs
    )

    print("正在训练决策树模型...")
    tree = DecisionTree(max_depth=max_depth)
    tree.fit(train_features, train_labels)

    print("正在评估模型...")
    error_rate = tree.auto_test(test_features, test_labels)
    acc = 1 - error_rate
    print(f"测试集错误率: {error_rate:.2%}")
    print(f"测试集准确率: {acc:.2%}")

    print("正在绘制二维分布图...")
    x_idx, y_idx = (0, 1) if n_features >= 2 else (0, 0)
    tree.plot_2d(features=features, labels=labels, x_idx=x_idx, y_idx=y_idx,
                 title=f"{name} ({n_samples} samples, {n_classes} classes)")
    return tree, acc


def test4():
    print("--- 开始运行决策树测试 ---")

    # 原始小数据集：隐形眼镜（24条，4特征，3分类）
    run_tree_test("隐形眼镜", "lenses.txt", max_depth=5)

    # 合成数据集1：简单二分类（150条，2特征）
    run_tree_test("合成数据-简单二分类", "synthetic1.txt", max_depth=8)

    # 合成数据集2：三分类（150条，3特征）
    run_tree_test("合成数据-三分类", "synthetic2.txt", max_depth=8)

    # 合成数据集3：带噪声二分类（150条，2特征）
    run_tree_test("合成数据-带噪声二分类", "synthetic3.txt", max_depth=8)


if __name__ == "__main__":
    # test1() 
    # test2()
    # test3()  
    test4()