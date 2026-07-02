# 学生成绩预测系统

基于机器学习的高校学生期末成绩预测系统，包含**数据层、算法层、应用层**三层架构。支持个体成绩预测、总体特征分析和决策树模型可视化。

## 技术栈

| 层 | 技术 |
|----|------|
| 前端 | Vue 3 + Vite + ECharts + Axios |
| 中间层 | Spring Boot 4.1 + Spring Data JPA |
| 模型层 | Python 3 + FastAPI + scikit-learn |
| 数据库 | MySQL 8.0 |

## 快速启动

### 环境要求

- Python ≥ 3.10
- Java JDK 17
- Node.js ≥ 22.18
- MySQL 8.0（root 密码 `root123456`，或设置环境变量 `DB_PASSWORD`）

### 一键启动

```bash
# Windows: 双击运行
start.bat

# 或分步启动
cd score-predict-model
python main.py              # 1. 训练模型（首次运行）
python api.py               # 2. Python API → :5000

cd ../score-predict-backend
mvn spring-boot:run         # 3. Java 后端 → :8080

cd ../score-predict-frontend/student_score_predict
npm install && npm run dev  # 4. 前端 → :3000
```

### 数据导入（首次使用）

```bash
cd score-predict-model
python import_to_db.py      # 将 Excel 7 页数据导入 MySQL（518 条）
```

### 命令行交互预测

```bash
cd score-predict-model
python Interact.py          # 加载模型，逐项输入特征值，输出预测结果
```

启动后访问 **http://localhost:3000**。

## 服务端口

| 服务 | 地址 | 说明 |
|------|------|------|
| 前端 | http://localhost:3000 | Vue 开发服务器 |
| Java 后端 | http://localhost:8080 | REST API 入口 |
| Python API | http://localhost:5000 | 模型推理服务（仅 localhost） |
| MySQL | localhost:3306 | 数据库 |

## 项目结构

```
project/
├── pom.xml                     # Maven 聚合父 POM
├── start.bat                   # 一键启动脚本
├── README.md
├── 学生成绩预测系统设计文档.md    # 详细设计文档
├── score-predict-model/        # Python 模型层
│   ├── main.py                 # 离线训练入口
│   ├── api.py                  # FastAPI 推理服务
│   ├── Data.py                 # 数据清洗与特征工程
│   ├── Algorithm.py            # 双模型训练与评估
│   ├── Interact.py             # 命令行交互预测
│   ├── import_to_db.py         # Excel → MySQL 导入
│   ├── data/Score_dataset.xlsx # 原始数据集
│   └── saved_models/           # 训练产物
├── score-predict-backend/      # Java 中间层
│   └── src/main/java/com/huahuo/demo/
│       ├── controller/         # REST 控制器
│       ├── service/            # 业务逻辑
│       ├── entity/             # JPA 实体
│       └── repository/         # 数据访问层
└── score-predict-frontend/     # Vue 前端
    └── student_score_predict/src/
        ├── views/
        │   ├── PredictView.vue # 个体预测页
        │   ├── AnalysisView.vue# 总体分析页
        │   ├── TrainView.vue   # 模型训练页
        │   ├── TreeView.vue    # 决策树可视化页
        │   └── HistoryView.vue # 历史数据页
        ├── api/index.js        # API 封装
        └── router/index.js     # 路由配置
```

## 功能

### 个体预测
教师通过滑块输入学生 5 项指标（线下互动、综合平时成绩、期末总成绩、平时成绩、期末成绩），系统实时返回预测分数和成绩等级（优/良/中/不及格）。

### 总体分析
- **特征重要性柱状图**：展示决策树模型各特征权重
- **相关性热力图**：Pearson 相关系数矩阵可视化

### 模型可视化
以树形图渲染分类决策树完整结构，展示每个分支节点的判定条件和叶子节点的分类结果。

### 模型训练
在线交互式训练界面，支持自定义选择参与训练的特征列，调节测试集比例（10%-40%）、决策树最大深度（2-15）、随机种子、相关性阈值等超参数。训练完成后展示线性回归和决策树两个模型的完整评估指标（R²/MAE/RMSE/Accuracy/Precision/Recall）、混淆矩阵、特征权重及特征重要性图表。

### 历史数据
展示已导入的 518 条学生历史成绩数据，提供统计卡片（总人数、四项均值）和成绩分布柱状图，支持分页浏览完整数据表。

## API 接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/prediction/features` | GET | 获取特征名与分类标签 |
| `/api/prediction/predict` | POST | 提交特征值，返回预测分数和等级 |
| `/api/prediction/analysis` | GET | 获取特征重要性 + 相关性矩阵 |
| `/api/prediction/tree` | GET | 获取决策树结构（ECharts 格式） |
| `/api/prediction/train/options` | GET | 获取可训练特征列表 |
| `/api/prediction/train` | POST | 提交训练配置，触发模型训练 |
| `/api/students/history` | GET | 获取历史学生数据 |

### 预测请求示例

```json
POST /api/prediction/predict
Content-Type: application/json

{ "features": [85, 77, 77, 89, 67] }

Response:
{
  "code": 200,
  "data": {
    "predicted_score": 92.64,
    "predicted_label": "优"
  }
}
```

## 模型性能

| 模型 | 指标 | 值 |
|------|------|-----|
| 多元线性回归 | R² | 0.8757 |
| 多元线性回归 | MAE | 3.50 |
| 分类决策树 | Accuracy | 66.67% |
| 分类决策树 | Precision (weighted) | 1.0000 |
| 分类决策树 | Recall (weighted) | 0.6667 |

## 设计文档

详见 [学生成绩预测系统设计文档.md](学生成绩预测系统设计文档.md)。
