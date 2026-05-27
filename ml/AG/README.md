# ML-Algorithms

从零实现经典机器学习算法，基于 NumPy，不依赖 `sklearn` 内置模型。

## 项目架构

```
BaseAlgorithm (ABC, MLVisualizerMixin)
├── KNN          - K 近邻
├── Logistic     - 逻辑回归
├── Regression   - 线性回归
└── DecisionTree - 决策树（ID3 风格）
```

所有算法继承自 `BaseAlgorithm`（`BaseAlgorithm.py`），统一实现 `fit()` 和 `predict()` 接口，并复用 `auto_test()` 和 `plot_2d()` 等通用方法。

## 环境准备

```bash
pip install numpy matplotlib pandas scikit-learn
```

## 算法列表

| 算法 | 文件 | 核心思想 |
|------|------|---------|
| KNN | `knn.py` | 计算测试样本与所有训练样本的欧氏距离，取前 K 个最近邻居的多数票作为预测结果 |
| 逻辑回归 | `Logistic.py` | 用 Sigmoid 将线性输出映射为概率，通过梯度下降迭代最小化交叉熵损失 |
| 线性回归 | `Regression.py` | 最小化预测值与真实值之间的均方误差 (MSE)，使用梯度下降求解 |
| 决策树 | `Decision_Tree.py` | 基于信息增益递归选择最优特征和阈值进行分裂，构建树形分类模型 |

## KNN

**类**：`KNN(BaseAlgorithm)`，参数 `k=20`。

**训练**：惰性学习，`fit()` 直接存储全部训练数据，无显式计算。
**预测**：计算输入样本与所有训练样本的欧氏距离 → `argsort()` 排序取前 K 个 → 多数投票返回类别。

## 逻辑回归

**类**：`Logistic(BaseAlgorithm)`，参数 `lr=0.01`，`iters=1000`。

**训练**：初始化 `w`（零向量）和 `b`（0）→ 每轮计算 `z = wᵀX + b` → Sigmoid 映射为概率 `a` → 误差 `a - y` → 梯度下降更新 `w`、`b`。
**预测**：`z = wᵀx + b` → Sigmoid → 概率 ≥ 0.5 判为 1，否则为 0。

梯度：`dw = (1/n) · Xᵀ · (a - y)`，`db = (1/n) · Σ(a - y)`。

## 线性回归

**类**：`Regression(BaseAlgorithm)`，参数 `lr=0.01`，`iters=1000`。

**训练**：初始化 `w` 和 `b` → 每轮计算预测值 `z = wᵀX + b` → 误差 `z - y` → 梯度下降更新。
**预测**：直接返回连续值 `wᵀx + b`。
**评估**：重写 `auto_test()`，返回 **MSE（均方误差）** 而非分类错误率。

附加 `plot_regression_line()` 方法：绘制散点图 + 红色拟合直线。

## 决策树

### 核心结构

- **`Node`**：树的节点，存储 `feature`（划分特征索引）、`threshold`（划分阈值）、`left`/`right`（左右子树指针）、`value`（叶节点类别值）。
- **`DecisionTree`**：继承自 `BaseAlgorithm`，通过三个超参数控制树的复杂度。

### 超参数

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `min_samples_split` | 节点再分裂所需的最小样本数 | 2 |
| `min_samples_leaf` | 叶节点最少样本数 | 1 |
| `max_depth` | 树的最大深度 | 100 |

### 训练流程 (`_build_tree`)

1. **检查终止条件**：当前深度达到最大深度、所有样本同一类别、或样本数不足 `min_samples_split`，则返回叶节点（取当前多数类）。
2. **寻找最佳分裂点** (`_best_split`)：枚举所有特征和唯一取值作为候选阈值，对每个 (特征, 阈值) 组合计算信息增益，选出增益最大的组合。
3. **验证分裂有效性**：若左右子节点任一不满足 `min_samples_leaf` 要求，放弃本次分裂，返回叶节点。
4. **递归构建子树**：按最佳分裂点将数据一分为二，分别递归建树。

### 信息增益 (`_information_gain`)

- **信息熵** (`_entropy`)：`-Σ p_i · log₂(p_i)`。全为同一类时熵为 0，各类均匀分布时熵最大。
- **信息增益** = 父节点熵 - 加权子节点熵。增益越大说明该分裂越有效。

### 预测流程 (`_traverse_tree`)

从根节点出发，根据每个节点的 `feature` 和 `threshold` 判断：当前特征值 ≤ 阈值走左子树，否则走右子树，直到到达叶节点，返回叶节点的 `value`。

## 使用示例

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

## 目录结构

```
AG/
├── BaseAlgorithm.py      # 抽象基类 + 可视化 Mixin
├── Decision_Tree.py      # 决策树（ID3 风格，含 Node 类）
├── knn.py                # K 近邻
├── Logistic.py           # 逻辑回归
├── Regression.py         # 线性回归
├── main.py               # 测试主入口
├── hw.py                 # 线性回归独立测试（鸢尾花数据集）
├── DecisionTree/         # 决策树数据集（隐形眼镜 + 3 个合成数据）
├── KNN/data/             # KNN 数据集（约会数据）
├── LOGISTIC/             # 逻辑回归数据集（马疝病数据）
└── REGRESSION/           # 回归数据集（鲍鱼年龄数据）
```

## 运行测试

```bash
python main.py
```

当前 `main.py` 默认运行 `test4()`（决策树测试），可取消注释 `test1()` ~ `test3()` 测试其他算法：

| 测试函数 | 算法 | 数据集 |
|----------|------|--------|
| `test1()` | KNN | 约会数据 (datingTestSet2.txt) |
| `test2()` | 逻辑回归 | 马疝病数据 (HorseColic) |
| `test3()` | 线性回归 | 鲍鱼年龄数据 (abalone.txt) |
| `test4()` | 决策树 | 隐形眼镜 + 3 个合成数据集 |

`test2()` 和 `test3()` 使用 `@get_min_error_rate` 装饰器自动进行网格搜索，寻找最佳学习率和迭代次数。

### 辅助测试

```bash
python hw.py
```

使用鸢尾花数据集进行线性回归独立测试，输出 R² 和 MSE 评估指标。
