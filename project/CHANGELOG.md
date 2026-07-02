# 变更日志

学生成绩预测系统的所有重要变更。

格式遵循 [Keep a Changelog](https://keepachangelog.com/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

---

## [Unreleased]

### 新增
- **手动录入学生数据**：`HistoryView.vue` 新增"添加学生"模态表单，支持逐条录入
  学生信息（9 个字段），`StudentController` 新增 `POST /api/students` 和
  `DELETE /api/students/{id}` 端点。
- **数据库数据源训练**：`TrainView.vue` 新增 Excel/数据库数据源切换，
  `main.py` 新增 `train_model_from_db()` 从 MySQL 读取数据进行训练，
  `api.py` `/train` 和 `/train/options` 端点支持 `data_source` 参数。
- `import_to_db.py` 新增 `线下_期末考试`、`线下总成绩` 两列导入，
  实现 DB 与 Excel 训练特征集一致。
- `StudentEntity` 新增 `offlineFinalExam`、`offlineTotal` 字段。
- **Ridge 回归 + 自动共线性检测**：`Algorithm.py` 将 `LinearRegression` 替换为
  `Ridge(alpha=1.0)`，L2 正则化压制多重共线性，权重分配更均衡。
  `_train_from_dataframe()` 自动检测特征间 |r| > 0.95 的高相关对并剔除冗余特征。

### 修复
- **DB 训练 NaN 错误**：`train_model_from_db()` 中 MySQL NULL 值导致 pandas
  将列类型设为 `object`，`select_dtypes` 无法检测数值列，缺失值未被填充即进入
  训练。改为先用 `pd.to_numeric()` 强制转换所有列为数值类型，再按均值/众数填充。
- **模型权重失衡**：原 `LinearRegression` 在多重共线性下权重不稳定，期末成绩
  权重 1.14、平时成绩权重 1.0 远超其他特征。Ridge 回归将期末成绩权重降至 0.76
  （-33%）、平时成绩降至 0.62（-38%），特征间影响更均衡。

### 变更
- `main.py`：抽取 `_train_from_dataframe()` 共享训练管线，新增自动共线性检测。
- `Algorithm.py`：`LinearRegression` → `Ridge(alpha=1.0)`。
- `StudentController` / `StudentService`：新增学生记录的增删功能。
- `PredictionController` / `PredictionService`：`trainOptions()` 新增
  `source` 参数传递数据源类型。
- `api/index.js`：新增 `addStudent()`、`deleteStudent()` 函数，
  `getTrainOptions()` 支持 `source` 参数。

---

## [0.3.0] — 2026-07-01

### 新增
- `pom.xml` — Maven 项目配置（Spring Boot 3.x + Spring Data JPA + MySQL
  Connector）
- `StudentRepository.java` — `JpaRepository<StudentEntity, Long>`
- `Application.java` — Spring Boot 启动入口，含 `CommandLineRunner`
- 前端工程脚手架（Vue 3 + Vite + Vue Router + Axios）：
  - `App.vue` — 根布局与导航
  - `PredictView.vue` — 预测表单 → 结果卡片
  - `AnalysisView.vue` — 特征重要性图 + 相关性热力图
  - `HistoryView.vue` — 历史预测记录表
  - `TrainView.vue` — 模型训练面板（占位）
  - `router/index.js` — 前端路由
  - `api/index.js` — Axios 实例与接口封装
- `学生成绩预测系统设计文档.md` — 架构图、接口规范、数据流、部署说明
  （工业级文档格式）
- `README.md` — 快速开始、环境要求、常见问题
- `requirements.txt` — Python 依赖清单

### 变更
- `PredictionController.java`：响应格式统一包装为 `{code, message, data}`
  信封，供前端统一解析。
- `PredictionService.java`：`RestTemplate` POST 逻辑抽取为可复用方法；
  增加错误响应反序列化。
- `StudentController.java`：增加分页查询参数（`page`、`size`）。
- `application.properties`：补充 `spring.datasource.*` MySQL 配置，
  `server.port=8080`。

---

## [0.2.0] — 2026-06-29

### 新增
- `PredictionController.java` — `POST /api/prediction/predict`
- `PredictionService.java` — 调用 Python `POST /predict`
- `PredictionRequest.java` — `{attendance, homework, midterm, participation,
  ...}` 请求体，含 Bean Validation
- `StudentController.java` — `GET/POST/PUT/DELETE /api/students`
- `StudentService.java` — 业务逻辑层，含 `@Transactional`
- `StudentEntity.java` — `@Entity` 映射到 `t_student` 表
- `application.yml` — 结构化配置（数据源、JPA、服务器）
- `application.properties` — 扁平配置兜底

### 修复
- **数据泄漏**（同 Unreleased —— 同一根因，首次修复作用于 `main.py`
  第 28–29 行附近）。
- `Score_dataset.xlsx`：3 层合并表头替换为单行标准化表头；公式单元格
  转为计算值；Sheet6 第 1–69 行数据完整保留。

### 变更
- `Data.py`：`load_data()` 适配新列名。
- `main.py`：目标列映射更新；`train_test_split` 前插入泄漏特征过滤器。

---

## [0.1.0] — 2026-06-28

### 新增
- `Algorithm.py` — 决策树回归管道（训练、预测、评估、交叉验证、特征重要性）
- `Data.py` — Excel 读取、缺失值填充、异常值检测、独热编码、特征-目标分离
- `Interact.py` — matplotlib 可视化（散点图、残差图、相关性热力图）
- `main.py` — CLI 入口，编排 加载→预处理→训练→评估→可视化 全流程
- `data/Score_dataset.xlsx` — 原始数据集（7 页，每页 69 名学生）

---

[Unreleased]: https://github.com/huahuo/score-predict/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/huahuo/score-predict/compare/v0.2.0...v0.3.0

[0.2.0]: https://github.com/huahuo/score-predict/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/huahuo/score-predict/releases/tag/v0.1.0
