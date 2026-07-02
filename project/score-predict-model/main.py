import os
import sys
import json
import argparse
import joblib
import pandas as pd
import numpy as np
import pymysql
from sklearn.model_selection import train_test_split
from Data import Data
from Algorithm import Algorithm

base_path = os.path.dirname(__file__)
input_path = os.path.join(base_path, 'data', 'Score_dataset.xlsx')
output_path = os.path.join(base_path, 'data_show', 'result.json')

DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "root123456",
    "database": "student_predict",
    "charset": "utf8mb4",
}

# DB 列名 → Excel 中文列名
DB_TO_EXCEL_MAP = {
    "interaction": "线下_互动",
    "offline_final_exam": "线下_期末考试",
    "offline_total": "线下总成绩",
    "comprehensive_regular": "综合_平时成绩",
    "final_total": "期末总成绩",
    "regular_score": "平时成绩",
    "final_score": "期末成绩",
    "online_total": "线上总成绩",
}

DEFAULT_CONFIG = {
    "target_col": "线上总成绩",
    "test_size": 0.2,
    "random_state": 42,
    "max_depth": 5,
    "correlation_threshold": 0.1,
    "range_limit": {
        "线上_平时成绩": [0, 40],
        "线上_期中测试": [0, 10],
        "线上_期末考试": [0, 40],
        "线上总成绩": [0, 100]
    },
    "feature_names": None,
    "class_bins": [-1, 59.9, 79.9, 89.9, 101],
    "class_labels": ["不及格", "中", "良", "优"]
}


def _train_from_dataframe(df: pd.DataFrame, config: dict) -> dict:
    """从已加载的 DataFrame 执行完整的训练管线（独热编码→特征选择→训练→评估→保存）"""
    cfg = {**DEFAULT_CONFIG, **config}
    data = Data(input_path, output_path)

    # ---- 独热编码 ----
    if "线下_互动" in df.columns:
        print("\n--- 独热编码 (One-Hot Encoding) ---")
        bins_interact = [-1, 40, 70, 101]
        labels_interact = ["低参与度", "中参与度", "高参与度"]
        df["参与度等级"] = pd.cut(df["线下_互动"], bins=bins_interact, labels=labels_interact)
        print(f"类别分布: {df['参与度等级'].value_counts().to_dict()}")

        feature_limit = ["参与度等级"]
        dummies = pd.get_dummies(df[feature_limit], prefix=feature_limit, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        df.drop(columns=feature_limit, inplace=True)
        print(f"独热编码后新增列: {list(dummies.columns)}")

    # ---- 特征选择 ----
    target_col = cfg["target_col"]

    if cfg["feature_names"] is not None:
        selected = list(cfg["feature_names"])
        if "线下_互动" in selected:
            onehot_cols = [c for c in df.columns if c.startswith("参与度等级_")]
            for oh in onehot_cols:
                if oh not in selected:
                    selected.append(oh)
            print(f"检测到 线下_互动，自动补全独热编码列: {onehot_cols}")
        print(f"\n使用手动指定的特征: {selected}")
        available = [c for c in selected if c in df.columns]
        missing = [c for c in selected if c not in df.columns]
        if missing:
            print(f"警告: 以下特征在数据中不存在，已忽略: {missing}")
        final_df = df[available + [target_col]]
    else:
        print("\n开始自动特征选择...")
        final_df = data.feature_choose(df, target_col, cfg["correlation_threshold"])

    # 剔除已知共线特征（数据泄漏）
    leakage_cols = ["线上_平时成绩", "线上_期中测试", "线上_期末考试"]
    existing_leakage = [c for c in leakage_cols if c in final_df.columns]
    if existing_leakage:
        print(f"\n剔除共线特征: {existing_leakage}")
        final_df = final_df.drop(columns=existing_leakage)

    # 自动检测并剔除高相关特征对（|r| > 0.95），保留与目标更相关的那个
    feature_cols = [c for c in final_df.columns if c != target_col]
    if len(feature_cols) >= 2:
        feature_corr = final_df[feature_cols].corr().abs()
        to_drop = set()
        for i in range(len(feature_cols)):
            for j in range(i + 1, len(feature_cols)):
                if feature_corr.iloc[i, j] > 0.95:
                    a, b = feature_cols[i], feature_cols[j]
                    if a in to_drop or b in to_drop:
                        continue
                    corr_a = abs(final_df[a].corr(final_df[target_col]))
                    corr_b = abs(final_df[b].corr(final_df[target_col]))
                    drop = a if corr_a <= corr_b else b
                    to_drop.add(drop)
                    print(f"剔除高相关特征: {drop} (与 {a if drop != a else b} 相关系数 {feature_corr.iloc[i, j]:.4f})")
        if to_drop:
            final_df = final_df.drop(columns=list(to_drop))

    print(f"\n最终使用的特征 ({len(final_df.columns) - 1} 个): {[c for c in final_df.columns if c != target_col]}")

    # ---- 准备训练数据 ----
    X = final_df.drop(columns=[target_col])
    y_reg = final_df[target_col]

    bins = cfg["class_bins"]
    labels = cfg["class_labels"]
    y_clf = pd.cut(y_reg, bins=bins, labels=labels)

    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = train_test_split(
        X, y_reg, y_clf, test_size=cfg["test_size"], random_state=cfg["random_state"]
    )

    # ---- 算法层 ----
    print("\n开始训练模型...")
    algo = Algorithm(max_depth=cfg["max_depth"], random_state=cfg["random_state"])
    algo.fit(X_train, y_train_reg, y_train_clf)

    lr_metrics = algo.LinearRegression_assessment(X_test, y_test_reg)
    tree_metrics = algo.tree_assessment(X_test, y_test_clf)

    # ---- 保存模型 ----
    model_dir = os.path.join(base_path, "saved_models")
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(algo.lr, os.path.join(model_dir, "lr_model.pkl"))
    joblib.dump(algo.clf, os.path.join(model_dir, "clf_model.pkl"))

    feature_names = algo.feature_names if algo.feature_names else X.columns.tolist()
    metadata = {
        "feature_names": feature_names,
        "target_col": target_col,
        "class_labels": labels
    }
    with open(os.path.join(model_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n模型已保存到: {model_dir}")

    # ---- 计算特征重要性 ----
    feature_importance = []
    if hasattr(algo.clf, "feature_importances_"):
        importances = algo.clf.feature_importances_.tolist()
        feature_importance = [
            {"name": n, "value": round(v, 4)}
            for n, v in zip(feature_names, importances)
        ]

    # ---- 计算相关性矩阵 ----
    numeric_df = final_df.select_dtypes(include=[np.number])
    corr_cols = [c for c in feature_names if c in numeric_df.columns]
    correlation_matrix = []
    if len(corr_cols) >= 2:
        corr = numeric_df[corr_cols].corr().round(4)
        correlation_matrix = corr.values.tolist()

    return {
        "feature_names": feature_names,
        "target_col": target_col,
        "class_labels": labels,
        "metrics": {
            "linear_regression": lr_metrics,
            "decision_tree": tree_metrics
        },
        "feature_importance": feature_importance,
        "correlation_matrix": correlation_matrix,
        "correlation_labels": corr_cols if correlation_matrix else feature_names,
        "config": cfg
    }


def train_model(config: dict = None) -> dict:
    """从 Excel 文件训练模型"""
    if config is None:
        config = {}
    cfg = {**DEFAULT_CONFIG, **config}

    data = Data(input_path, output_path)
    print("开始清洗数据...")
    df = data.data_show(cfg["range_limit"])

    return _train_from_dataframe(df, cfg)


def train_model_from_db(config: dict = None) -> dict:
    """从 MySQL 数据库读取数据训练模型"""
    if config is None:
        config = {}
    cfg = {**DEFAULT_CONFIG, **config}

    conn = pymysql.connect(**DB_CONFIG)
    try:
        df = pd.read_sql("SELECT * FROM t_student_history", conn)
    finally:
        conn.close()

    if df.empty:
        raise ValueError("数据库 t_student_history 表为空，请先导入或手动录入数据")

    # 去掉 id 和姓名列
    df = df.drop(columns=[c for c in ["id", "student_name"] if c in df.columns])

    # DB 列名映射为中文
    df = df.rename(columns=DB_TO_EXCEL_MAP)

    # 强制所有列转为数值类型（MySQL NULL 会使列变成 object，必须先转）
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # 填充缺失值：数值列用均值，文本列用众数（与 Data.data_show 一致）
    for c in df.columns:
        if df[c].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].mean())
        else:
            mode_vals = df[c].mode()
            df[c] = df[c].fillna(mode_vals[0] if not mode_vals.empty else '缺失')

    # 应用 range_limit 过滤（仅对存在的列）
    for col, (lo, hi) in cfg["range_limit"].items():
        if col in df.columns:
            df = df[(df[col] >= lo) & (df[col] <= hi)]

    # 写入 result.json 供 /api/analysis 使用
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_json(output_path, orient="records", force_ascii=False)

    print(f"从数据库读取 {len(df)} 条数据，列: {list(df.columns)}")
    return _train_from_dataframe(df, cfg)


def get_available_features() -> dict:
    """返回 Excel 数据集中所有可用的数值特征列表"""
    df = pd.read_excel(input_path)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    exclude = ["线上_平时成绩", "线上_期中测试", "线上_期末考试"]
    selectable = [c for c in numeric_cols if c not in exclude]

    categorical_hints = []
    if "线下_互动" in df.columns:
        categorical_hints.append({
            "source": "线下_互动",
            "onehot_labels": ["参与度等级_低参与度", "参与度等级_中参与度", "参与度等级_高参与度"],
            "description": "根据线下互动分数自动生成的参与度等级"
        })

    return {
        "numeric_features": selectable,
        "categorical_features": categorical_hints,
        "default_target": "线上总成绩",
        "possible_targets": [c for c in numeric_cols if c not in exclude or c == "线上总成绩"],
        "total_samples": len(df)
    }


def get_available_features_from_db() -> dict:
    """返回数据库中所有可用的数值特征列表"""
    conn = pymysql.connect(**DB_CONFIG)
    try:
        df = pd.read_sql("SELECT * FROM t_student_history", conn)
    finally:
        conn.close()

    if df.empty:
        return {
            "numeric_features": [],
            "categorical_features": [],
            "default_target": "线上总成绩",
            "possible_targets": ["线上总成绩"],
            "total_samples": 0
        }

    df = df.drop(columns=[c for c in ["id", "student_name"] if c in df.columns])
    df = df.rename(columns=DB_TO_EXCEL_MAP)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    exclude = ["线上_平时成绩", "线上_期中测试", "线上_期末考试"]
    selectable = [c for c in numeric_cols if c not in exclude]

    categorical_hints = [{
        "source": "线下_互动",
        "onehot_labels": ["参与度等级_低参与度", "参与度等级_中参与度", "参与度等级_高参与度"],
        "description": "根据线下互动分数自动生成的参与度等级"
    }] if "线下_互动" in df.columns else []

    return {
        "numeric_features": selectable,
        "categorical_features": categorical_hints,
        "default_target": "线上总成绩",
        "possible_targets": [c for c in numeric_cols if c not in exclude or c == "线上总成绩"],
        "total_samples": len(df)
    }


def main():
    parser = argparse.ArgumentParser(description="学生成绩预测模型训练")
    parser.add_argument("--target", default=None, help="目标列名")
    parser.add_argument("--features", nargs="*", default=None, help="手动指定特征列名")
    parser.add_argument("--test-size", type=float, default=None, help="测试集比例")
    parser.add_argument("--max-depth", type=int, default=None, help="决策树最大深度")
    parser.add_argument("--random-state", type=int, default=None, help="随机种子")
    parser.add_argument("--correlation-threshold", type=float, default=None, help="特征选择相关性阈值")
    parser.add_argument("--show-features", action="store_true", help="仅显示可用特征列表")
    parser.add_argument("--source", default="excel", choices=["excel", "db"], help="数据源: excel 或 db")

    args = parser.parse_args()

    if args.show_features:
        if args.source == "db":
            info = get_available_features_from_db()
        else:
            info = get_available_features()
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return

    # 只把非 None 的参数传入 config
    config = {}
    if args.target is not None:
        config["target_col"] = args.target
    if args.features is not None:
        config["feature_names"] = args.features
    if args.test_size is not None:
        config["test_size"] = args.test_size
    if args.max_depth is not None:
        config["max_depth"] = args.max_depth
    if args.random_state is not None:
        config["random_state"] = args.random_state
    if args.correlation_threshold is not None:
        config["correlation_threshold"] = args.correlation_threshold

    result = train_model_from_db(config if config else None) if args.source == "db" else train_model(config if config else None)

    print("\n" + "=" * 50)
    print("  训练完成 - 结果摘要")
    print("=" * 50)
    print(f"使用特征: {result['feature_names']}")
    print(f"线性回归 R²: {result['metrics']['linear_regression']['r2']:.4f}")
    print(f"决策树准确率: {result['metrics']['decision_tree']['accuracy'] * 100:.2f}%")
    print(f"特征重要性: {result['feature_importance']}")


if __name__ == "__main__":
    main()
