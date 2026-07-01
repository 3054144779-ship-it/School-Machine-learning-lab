# 变更日志

学生成绩预测系统的所有重要变更。

格式遵循 [Keep a Changelog](https://keepachangelog.com/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

---

## [Unreleased]

### 新增

- `TrainView.vue` — 模型训练界面，支持动态选择训练特征与超参数调优
  （最大深度、最小样本分裂数、测试集比例）
- `api.py` 新增 `POST /train` 端点 — 接收特征列表与超参数，返回结构化评估指标
- `api.py` 新增 `GET /features/available` 端点 — 暴露所有可训练的特征列
- `api.py` 新增 `GET /evaluate` 端点 — 返回当前模型评估指标 JSON
- `import_to_db.py` — 将 `Score_dataset.xlsx` 全部 7 页数据通过 SQLAlchemy
  批量导入 MySQL
- `start.bat` — 一键启动脚本（依次拉起 Python API → Spring Boot → Vite 开发服务器）

### 修复
- **数据泄漏**：`main.py` 将 `线上_平时成绩`、`线上_期中测试`、`线上_期末考试`
  作为特征，而目标是 `线上总成绩`。三者求和恰好等于目标，导致 R² 虚高至 1.0。
  现于特征工程阶段剔除与目标共线的子项。R² 1.0→0.8793，MAE 0→3.46。
- **特征维度不匹配**：`reload_models()` 对模型文件与元数据的更新非原子操作，
  导致 `DISPLAY_FEATURES` 推导出 N 列而 pickle 模型期望 M 列。
  改为先写临时文件再原子重命名，加载后追加维度一致性断言。
- **预测静默失败**：`PredictView.vue` 检测到维度不匹配后未重试预测，
  用户点击后毫无反馈。现改为自动重载特征后静默重试一次。
- **独热编码遗漏**：训练时选择类别型特征 `线下_互动` 未自动包含其独热编码
  派生列，导致维度不匹配。已在 `/train` 处理逻辑中修复。
- `PredictionService.java` 现在透传 Python API 的错误体到前端日志，
  不再吞掉异常信息。
- `RestTemplateConfig.java` 增加读写超时配置，防止长时间训练导致连接假死。
- `HistoryView.vue` 本地存在但未被 git 追踪，导致其他机器克隆后编译失败
  （对应提交 `e810804`）。

### 变更
- `Algorithm.py`：评估方法（`evaluate_model`、`cross_validate`）改为返回
  结构化 dict，不再直接打印到 stdout。
- `api.py`：`reload_models()` 改为原子操作 —— 先写入临时文件再重命名，
  加载后执行维度一致性校验。
- `PredictView.vue`：特征维度处理逻辑抽取为公共函数 `loadFeatures()`；
  维度不匹配时自动重试预测。
- `Data.py` / `main.py`：列名适配清洗后的 Excel 表头
  （`线上_平时成绩`、`线下_互动` 等）。

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
