# AG (Algorithm) 机器学习算法框架技术文档

## 1. 模块简介

本项目（AG）是一个从零实现的经典机器学习算法框架。为了规范各类机器学习算法（KNN、逻辑回归、线性回归、决策树等）的开发，项目采用面向对象的设计思想，提取了通用图表绘制与模型评估逻辑，封装在核心基类 `BaseAlgorithm.py` 中，为所有具体算法子类提供统一的接口规范。所有算法基于 NumPy 实现，不依赖 `sklearn` 内置模型。

## 2. 基础架构

项目采用"基类统一定义，子类分别实现"的工程架构：

```text
AG/
├── BaseAlgorithm.py       # 核心文件：抽象基类 + 可视化 Mixin
├── Decision_Tree.py       # 子类实现：决策树算法（ID3 风格）
├── knn.py                 # 子类实现：K 近邻算法
├── Logistic.py            # 子类实现：逻辑回归算法
├── Regression.py          # 子类实现：线性回归算法
├── hw.py                  # 辅助文件：线性回归独立测试
├── main.py                # 程序主入口：实例化各算法并执行测试
├── DecisionTree/          # 决策树数据集（隐形眼镜 + 合成数据）
├── KNN/data/              # KNN 数据集（约会数据）
├── LOGISTIC/              # 逻辑回归数据集（马疝病数据）
└── REGRESSION/            # 回归数据集（鲍鱼年龄数据）
```

### 类继承关系

```
BaseAlgorithm (ABC, MLVisualizerMixin)
├── KNN          - K 近邻分类器
├── Logistic     - 逻辑回归二分类器
├── Regression   - 线性回归模型
└── DecisionTree - 决策树分类器（ID3 风格，含 Node 辅助类）
```

所有算法继承自 `BaseAlgorithm`，统一实现 `fit()` 和 `predict()` 接口，并复用 `auto_test()` 和 `plot_2d()` 等通用方法。

## 3. 环境准备

基类依赖 Python 的科学计算和数据可视化标准库：

```bash
pip install numpy matplotlib pandas scikit-learn
```

| 依赖 | 用途 |
|------|------|
| `numpy` | 矩阵运算、梯度计算 |
| `matplotlib` | 数据可视化 |
| `pandas` | 决策树数据清洗与加载 |
| `scikit-learn` | 数据划分 (`train_test_split`)、标签编码 (`LabelEncoder`)、评估指标 |

## 4. 核心类说明

`BaseAlgorithm.py` 中包含两个核心类，通过多重继承组合：

| 类名 | 职责描述 |
|------|---------|
| `MLVisualizerMixin` | **可视化混入类**：负责数据分布的散点图绘制，将 UI/画图逻辑与核心算法逻辑解耦。 |
| `BaseAlgorithm` | **算法抽象基类**：继承自 `ABC` 和 `MLVisualizerMixin`，强制规定所有具体 ML 算法的通用生命周期和评估标准。 |

## 5. 核心方法逻辑

### 5.1 可视化功能 (`MLVisualizerMixin`)

**`plot_2d(features, labels, x_idx=0, y_idx=0, title="text")`**

- 创建 8×6 尺寸画布。
- 提取特征矩阵 `features` 中指定的两列（`x_idx`, `y_idx`）作为坐标轴。
- 使用 `labels` 作为颜色映射（`cmap='viridis'`），绘制带黑色边缘的半透明散点。
- 自动添加颜色条（Colorbar）以区分不同类别。

### 5.2 算法抽象与评估 (`BaseAlgorithm`)

| 方法 | 类型 | 功能 |
|------|------|------|
| `__init__(self)` | 构造 | 初始化 `train_data` 和 `train_labels` 为 `None` |
| `fit(self, data, labels)` | **抽象方法** | 定义模型训练过程，子类必须重写 |
| `predict(self, input_data)` | **抽象方法** | 定义模型预测/推理过程，子类必须重写 |
| `auto_test(self, test_data, test_labels)` | 通用 | 遍历测试集逐条预测，比对真实标签，返回**错误率** |

`auto_test` 逻辑：循环调用 `self.predict(test_data[i])`，统计预测错误数，返回 `error_count / len(test_data)`。注意：`Regression` 子类重写了此方法，改为返回 **MSE（均方误差）**。

## 6. 各算法详解

### 6.1 K 近邻 (KNN) — `knn.py`

| 项目 | 说明 |
|------|------|
| 类名 | `KNN(BaseAlgorithm)` |
| 构造参数 | `k=20`（近邻数量） |
| 核心思想 | 计算测试样本与所有训练样本的欧氏距离，取前 K 个最近邻居的多数票作为预测结果 |
| 训练 (`fit`) | 直接存储训练数据和标签（惰性学习，无显式训练过程） |
| 预测 (`predict`) | 计算与所有训练样本的欧氏距离 → 排序取 Top-K → 多数投票 |

**计算流程**：`diffMat = train_data - input_data` → 平方求和 → 开方得距离 → `argsort()` 排序 → 统计前 K 个标签频次 → 返回最高频类别。

### 6.2 逻辑回归 (Logistic) — `Logistic.py`

| 项目 | 说明 |
|------|------|
| 类名 | `Logistic(BaseAlgorithm)` |
| 构造参数 | `lr=0.01`（学习率），`iters=1000`（迭代次数） |
| 核心思想 | 用 Sigmoid 将线性输出映射为概率，通过梯度下降最小化交叉熵损失 |
| 训练 (`fit`) | 初始化权重 `w`（零向量）和偏置 `b`（0）→ 每轮计算 `z = wᵀX + b` → Sigmoid 激活得概率 `a` → 计算误差 `a - y` → 求梯度更新 `w` 和 `b` |
| 预测 (`predict`) | 计算 `z` → Sigmoid → 概率 ≥ 0.5 判为 1，否则判为 0 |

**梯度公式**：`dw = (1/n) · Xᵀ · (a - y)`，`db = (1/n) · Σ(a - y)`。

### 6.3 线性回归 (Regression) — `Regression.py`

| 项目 | 说明 |
|------|------|
| 类名 | `Regression(BaseAlgorithm)` |
| 构造参数 | `lr=0.01`（学习率），`iters=1000`（迭代次数） |
| 核心思想 | 最小化预测值与真实值之间的均方误差 (MSE)，通过梯度下降求解最优参数 |
| 训练 (`fit`) | 初始化 `w` 和 `b` → 每轮计算预测值 `z = wᵀX + b` → 误差 `z - y` → 梯度下降更新 |
| 预测 (`predict`) | 直接返回 `wᵀX + b`（连续值） |
| 评估 (`auto_test`) | **重写**：返回 MSE 而非分类错误率 |

**附加方法**：`plot_regression_line(features, labels, x_idx=0)` — 绘制散点图及红色拟合直线。
**损失函数**：`_loss(z)` 返回 `(zᵀz) / (2m)`。

### 6.4 决策树 (DecisionTree) — `Decision_Tree.py`

| 项目 | 说明 |
|------|------|
| 类名 | `DecisionTree(BaseAlgorithm)` |
| 构造参数 | `min_samples_split=2`，`min_samples_leaf=1`，`max_depth=100` |
| 核心思想 | 基于信息增益递归选择最优特征和阈值进行分裂，构建树形分类模型 |
| 训练 (`fit`) | 调用 `_build_tree(data, labels)` 递归建树 |
| 预测 (`predict`) | 调用 `_traverse_tree(x, root)` 从根节点沿树走到叶节点 |

#### 决策树核心结构

**`Node` 类**：树的节点，存储划分特征索引 (`feature`)、划分阈值 (`threshold`)、左右子树指针 (`left` / `right`) 以及叶节点的类别值 (`value`)。通过 `is_leaf_node()` 判断是否为叶节点。

#### 超参数

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `min_samples_split` | 节点再分裂所需的最小样本数 | 2 |
| `min_samples_leaf` | 叶节点最少样本数 | 1 |
| `max_depth` | 树的最大深度 | 100 |

#### 训练流程 (`_build_tree`)

1. **检查终止条件**：深度达到 `max_depth`、所有样本同一类别、或样本数不足 `min_samples_split` → 返回叶节点（当前多数类）。
2. **寻找最佳分裂点** (`_best_split`)：枚举所有特征及其唯一取值作为候选阈值，对每个 (特征, 阈值) 组合计算信息增益，选出增益最大的组合。
3. **验证分裂有效性**：若左右子节点任一不满足 `min_samples_leaf` 要求，放弃本次分裂，返回叶节点。
4. **递归构建子树**：按最佳分裂点将数据一分为二，分别递归建树。

#### 信息增益 (`_information_gain`)

- **信息熵** (`_entropy`)：衡量数据混乱程度，公式为 `-Σ p_i · log₂(p_i)`。全为同一类时熵为 0，各类均匀分布时熵最大。
- **信息增益** = 父节点熵 - 加权子节点熵。增益越大说明该分裂越有效。

#### 预测流程 (`_traverse_tree`)

从根节点出发，若当前节点特征值 ≤ 阈值则走左子树，否则走右子树，直到到达叶节点，返回叶节点的 `value`。

## 7. 运行测试

```bash
python main.py
```

当前 `main.py` 默认运行 `test4()`（决策树测试）。可取消注释 `test1()` ~ `test3()` 测试其他算法：

| 测试函数 | 算法 | 数据集 | 说明 |
|----------|------|--------|------|
| `test1()` | KNN | 约会数据 (datingTestSet2.txt) | 3 特征分类，K=20，含归一化 |
| `test2()` | 逻辑回归 | 马疝病数据 (HorseColic) | 二分类，含网格搜索最佳 lr 和 iters |
| `test3()` | 线性回归 | 鲍鱼年龄数据 (abalone.txt) | 回归任务，含网格搜索，返回 MSE |
| `test4()` | 决策树 | 隐形眼镜 + 3 个合成数据集 | 含通用数据加载与清洗函数 |

### 辅助测试 (`hw.py`)

独立的线性回归测试，使用鸢尾花数据集（Iris），选取花瓣长度预测花萼长度，输出版 `R²` 和 `MSE` 评估指标，并绘制拟合线。

```bash
python hw.py
```

## 8. 开发与使用规范

- **强制约束**：`BaseAlgorithm` 是抽象基类（含有 `@abstractmethod`），**无法被直接实例化**。所有新增算法必须继承 `BaseAlgorithm` 并实现完整的 `fit` 和 `predict` 方法，否则 Python 解释器将报错。
- **功能复用**：子类实现完毕后，可直接调用 `self.plot_2d()` 展示数据分类效果，调用 `self.auto_test()` 快速验证算法准确性，无需重复造轮子。
- **评估差异**：分类算法（KNN、Logistic、DecisionTree）的 `auto_test` 返回错误率（0~1）；回归算法（Regression）的 `auto_test` 返回均方误差 (MSE)。
- **数据预处理**：各算法在 `main.py` 中均包含归一化/标准化步骤，KNN 和回归使用 min-max 归一化，Logistic 使用 Z-score 标准化。

## 9. 使用示例

```python
from Decision_Tree import DecisionTree
from sklearn.model_selection import train_test_split
import numpy as np

# 加载数据
data = np.loadtxt('your_dataset.txt')
features, labels = data[:, :-1], data[:, -1]

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=1
)

# 训练
tree = DecisionTree(max_depth=5)
tree.fit(X_train, y_train)

# 评估
error = tree.auto_test(X_test, y_test)
print(f"准确率: {1 - error:.2%}")

# 可视化
tree.plot_2d(features, labels, title="Decision Tree")
```

---
*文档版本：V2.0*
*开发者：[花火]*
