import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split # 数据集划分工具
from Data import Data
from Algorithm import Algorithm

# 获取当前文件位置
base_path = os.path.dirname(__file__)
# 获取处理表格路径和输出表格路径
input_path = os.path.join(base_path, 'data', 'Score_dataset.xlsx')
output_path = os.path.join(base_path, 'data_show', 'result.json')

def main():
    # 数据层
    data = Data(input_path, output_path)
    
    # 定义清洗范围
    range_limit = {
        "线上_平时成绩": [0, 40],
        "线上_期中测试": [0, 10],
        "线上_期末考试": [0, 40],
        "线上总成绩": [0, 100]
    }

    print("开始清洗数据...")
    df = data.data_show(range_limit)
    
    # ========== 独热编码实践 ==========
    # 将 "线下_互动" 数值列转换为类别特征 "参与度等级"，展示独热编码流程
    if "线下_互动" in df.columns:
        print("\n--- 独热编码 (One-Hot Encoding) ---")
        print("将 '线下_互动' 数值列转换为类别特征 '参与度等级' (低/中/高)")
        bins_interact = [-1, 40, 70, 101]
        labels_interact = ["低参与度", "中参与度", "高参与度"]
        df["参与度等级"] = pd.cut(df["线下_互动"], bins=bins_interact, labels=labels_interact)
        print(f"类别分布: {df['参与度等级'].value_counts().to_dict()}")

        # 对类别特征执行独热编码
        feature_limit = ["参与度等级"]
        dummies = pd.get_dummies(df[feature_limit], prefix=feature_limit)
        df = pd.concat([df, dummies], axis=1)
        df.drop(columns=feature_limit, inplace=True)
        print(f"独热编码后新增列: {list(dummies.columns)}")
    # ====================================

    print("\n开始特征选择...")
    # feature_choose 会返回筛选后的完整 DataFrame
    target_col = "线上总成绩"
    final_df = data.feature_choose(df, target_col)

    # 剔除与目标变量完全共线的子项特征（防止数据泄漏：线上总成绩 = 线上_平时成绩 + 线上_期中测试 + 线上_期末考试）
    leakage_cols = ["线上_平时成绩", "线上_期中测试", "线上_期末考试"]
    existing_leakage = [c for c in leakage_cols if c in final_df.columns]
    if existing_leakage:
        print(f"\n剔除共线特征: {existing_leakage}")
        final_df = final_df.drop(columns=existing_leakage)

    print("\n准备训练数据...")
    # X 是特征矩阵 (去掉目标列)
    X = final_df.drop(columns=[target_col])
    # y_reg 是连续数值目标 (供线性回归使用)
    y_reg = final_df[target_col]
    
    # 将连续分数转换为分类标签 
    # <60 不及格，60-79 中，80-89 良，90以上 优
    bins = [-1, 59.9, 79.9, 89.9, 101]
    labels = ["不及格", "中", "良", "优"]
    y_clf = pd.cut(y_reg, bins=bins, labels=labels)
    
    # 划分训练集和测试集 (80% 训练，20% 测试)
    # 把 X, y_reg, y_clf 绑定在一起同步打乱和划分
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42
    )

    # 算法层
    
    print("\n开始训练模型...")
    algo = Algorithm()

    # 训练模型 
    algo.fit(X_train, y_train_reg, y_train_clf) # 连续值 y 和 分类值 y
    
    # 模型评估
    # 评估模型 A：多元线性回归 
    algo.LinearRegression_assessment(X_test, y_test_reg)
    
    # 评估模型 B：分类决策树
    algo.tree_assessment(X_test, y_test_clf)

    # 保存模型和元数据，供 api.py 使用
    model_dir = os.path.join(base_path, "saved_models")
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(algo.lr, os.path.join(model_dir, "lr_model.pkl"))
    joblib.dump(algo.clf, os.path.join(model_dir, "clf_model.pkl"))

    metadata = {
        "feature_names": algo.feature_names if algo.feature_names else X.columns.tolist(),
        "target_col": target_col,
        "class_labels": labels
    }
    with open(os.path.join(model_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n模型已保存到: {model_dir}")

if __name__ == "__main__":
    main()