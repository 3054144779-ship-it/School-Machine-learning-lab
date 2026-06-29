import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from BaseAlgorithm import BaseAlgorithm

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature       # 划分特征的索引
        self.threshold = threshold   # 划分的阈值
        self.left = left             # 左子树 
        self.right = right           # 右子树 
        self.value = value           # 叶子节点，保存预测的类别

    # 判断是否为叶子节点
    def is_leaf_node(self):
        return self.value is not None

class DecisionTree(BaseAlgorithm):
    def __init__(self, min_samples_split=2, min_samples_leaf=1, max_depth=100):
        # 初始化父类，继承 train_data 和 train_labels 属性
        super().__init__()
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth # 最大深度
        self.root = None # 根节点

    def fit(self, data, labels):
        self.train_data = data
        self.train_labels = labels
        # 建树
        self.root = self._build_tree(data, labels)

    def predict(self, input_data):
        """
        注意: input_data 是一条单一样本 (1D array)
        """
        return self._traverse_tree(input_data, self.root)

    # --- 私有辅助方法部分 ---
    # 建树
    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        # base case
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # 寻找最佳分裂点
        best_feature, best_thresh = self._best_split(X, y, n_features)

        # 找不到有效分裂（如所有特征值相同）
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)

        # 任一子节点样本数不足 min_samples_leaf，放弃分裂
        if len(left_idxs) < self.min_samples_leaf or len(right_idxs) < self.min_samples_leaf:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_child = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_child = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left_child, right_child)

    # 枚举特征和阈值，找到最优。
    def _best_split(self, X, y, n_features):
        best_gain = -1
        split_idx, split_threshold = None, None
        for feature_idx in range(n_features):
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            for thr in thresholds:
                left_idxs, right_idxs = self._split(X_column, thr)
                if len(left_idxs) < self.min_samples_leaf or len(right_idxs) < self.min_samples_leaf:
                    continue
                gain = self._information_gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = thr
        return split_idx, split_threshold

    # 计算信息增益
    def _information_gain(self, y, X_column, threshold):
        '''
        切分前，计算一下当前数据的混乱程度（父节点信息熵）。
        切分后，计算左右两堆数据的平均混乱程度（子节点信息熵）。
        信息增益 = 之前的混乱程度 - 之后的混乱程度。混乱程度减少得越多，增益就越大。
        '''
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        return parent_entropy - child_entropy

    # 分割数据
    def _split(self, X_column, split_thresh):
        '''
        给它一列数据和一个阈值(比如花瓣长度 2.5)，它就把小于等于 2.5 的数据索引放左边，大于 2.5 的放右边，
        然后把这两拨人的“名单”（索引）交接给 _build_tree。
        '''
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    # 计算信息熵
    def _entropy(self, y):
        '''
        如果一堆数据全是同一类，熵就是 0(极端纯粹); 如果一半A一半B, 熵就是 1(极度混乱)。
        代码通过统计各类别比例 p, 代入公式 -xsum(plog_2(p)) 计算得出。
        '''
        hist = np.bincount(y.astype(int))
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    # 统计哪种类别居多
    def _most_common_label(self, y):
        if len(y) == 0:
            return 0
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    # 查询当前数据属于哪一类
    def _traverse_tree(self, x, node):
        """
        根据单条数据沿着树向下寻找叶子节点
        如果小于等于当前节点设定的规则, 向左, 否则向右
        """
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)