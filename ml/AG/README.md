# AG (Algorithm) 机器学习算法框架技术文档

## 1. 模块简介

本项目（AG）是一个基础的机器学习算法实战框架。为了规范各类机器学习算法（如 KNN、逻辑回归、决策树等）的开发，本项目采用面向对象的设计思想，提取了通用图表绘制与模型评估逻辑，封装在核心基类 `BaseAlgorithm.py` 中，为所有具体的算法子类提供统一的接口规范。

## 2. 基础架构
从项目目录结构可以看出，这是一个典型的“基类统一定义，子类分别实现”的工程架构：

```text
/AG ———|
       |——/DecisionTree, /KNN, /LOGISTIC, /REGRESSION # 各算法配套的数据或资源文件夹
       |
       |——BaseAlgorithm.py    # 核心文件：定义算法抽象基类与可视化 Mixin
       |
       |——Decision_Tree.py    # 子类实现：决策树算法
       |——knn.py              # 子类实现：K近邻算法
       |——Logistic.py         # 子类实现：逻辑回归算法
       |——Regression.py       # 子类实现：线性回归算法
       |
       |——main.py             # 程序主入口：负责实例化具体算法并执行测试
```

## 3. 环境准备
基类依赖 Python 的科学计算和数据可视化标准库。请确保安装以下依赖：

```bash
pip install numpy matplotlib
```

## 4. 核心类说明
`BaseAlgorithm.py` 文件中包含两个核心类，通过继承机制组合在一起：

| 类名                | 职责描述                                                     |
| :------------------ | :----------------------------------------------------------- |
| `MLVisualizerMixin` | **可视化混入类**：专门负责数据分布的图表绘制，将 UI/画图逻辑与核心算法逻辑解耦。 |
| `BaseAlgorithm`     | **算法抽象基类**：继承自 `ABC` (Abstract Base Class) 和 `MLVisualizerMixin`，强制规定了所有具体 ML 算法的通用生命周期和评估标准。 |

## 5. 核心方法逻辑

### 5.1 可视化功能 (`MLVisualizerMixin`)
1. **`plot_2d(features, labels, x_idx=0, y_idx=0, title="text")`**
   - **功能**：绘制 2D 数据散点图。
   - **逻辑**：创建一个 8x6 尺寸的画布，提取特征矩阵 `features` 中指定的两列（`x_idx`, `y_idx`）作为坐标轴，使用 `labels` 作为颜色映射（`cmap='viridis'`），绘制带有黑色边缘的半透明散点，并自动添加颜色条（Colorbar）以区分不同类别。

### 5.2 算法抽象与评估 (`BaseAlgorithm`)
1. **`__init__(self)`**
   - **功能**：初始化模型的基础属性 `train_data`（训练数据）和 `train_labels`（数据标签）。
2. **`fit(self, data, labels)`**
   - **功能**：**抽象方法**，定义模型的训练过程。任何继承此基类的具体算法（如 KNN）都**必须**重写此方法来实现具体的数学拟合逻辑。
3. **`predict(self, input_data)`**
   - **功能**：**抽象方法**，定义模型的预测/推理过程。子类必须实现此方法以接收输入并返回预测结果。
4. **`auto_test(self, test_data, test_labels)`**
   - **功能**：通用模型评估逻辑。
   - **逻辑**：接收测试集特征和真实标签，循环调用当前类的 `predict` 方法进行预测。通过比对预测结果 `res` 与真实标签 `test_labels[i]`，统计预测错误的样本数，最终返回模型的**错误率**（`error_count / len(test_data)`）。

## 6. 开发与使用规范
- **强制约束**：由于 `BaseAlgorithm` 是抽象基类（含有 `@abstractmethod`），**无法被直接实例化**。所有新增的算法（如编写在 `knn.py` 中）必须继承 `BaseAlgorithm` 并实现完整的 `fit` 和 `predict` 方法，否则 Python 解释器将报错。
- **功能复用**：子类实现完毕后，可以直接调用 `self.plot_2d()` 来展示数据分类效果，调用 `self.auto_test()` 来快速验证算法的准确性，无需重复造轮子。

---
*文档版本：V1.0*
*开发者：[花火]*