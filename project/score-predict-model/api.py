import os
import json
import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.tree import _tree

app = FastAPI(title="Student Score Predict API")

BASE_PATH = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_PATH, "saved_models")

# 加载模型
lr_model = None
clf_model = None
metadata = {}

try:
    lr_model = joblib.load(os.path.join(MODEL_DIR, "lr_model.pkl"))
    clf_model = joblib.load(os.path.join(MODEL_DIR, "clf_model.pkl"))
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"模型加载成功，特征: {metadata.get('feature_names', [])}")
except Exception as e:
    print(f"模型加载失败，请先运行 main.py 训练模型: {e}")

FEATURE_NAMES = metadata.get("feature_names", [])
CLASS_LABELS = metadata.get("class_labels", ["不及格", "中", "良", "优"])


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/api/features")
def get_features():
    """返回模型期望的特征名称和顺序"""
    if not FEATURE_NAMES:
        raise HTTPException(status_code=500, detail="模型未加载")
    return {
        "code": 200,
        "data": {
            "feature_names": FEATURE_NAMES,
            "class_labels": CLASS_LABELS
        }
    }


@app.post("/api/predict")
def predict_score(req: PredictRequest):
    if lr_model is None or clf_model is None:
        raise HTTPException(status_code=500, detail="模型未加载，请先运行 main.py 训练模型")
    if len(req.features) != len(FEATURE_NAMES):
        raise HTTPException(
            status_code=400,
            detail=f"特征数量不匹配，期望 {len(FEATURE_NAMES)} 个: {FEATURE_NAMES}"
        )

    X = np.array([req.features])

    # 线性回归预测连续分数
    score_reg = float(lr_model.predict(X)[0])
    # 决策树预测分类等级
    label_clf = str(clf_model.predict(X)[0])

    return {
        "code": 200,
        "message": "success",
        "data": {
            "predicted_score": round(score_reg, 2),
            "predicted_label": label_clf
        }
    }


@app.get("/api/analysis")
def get_analysis():
    if clf_model is None:
        raise HTTPException(status_code=500, detail="模型未加载")

    try:
        importances = clf_model.feature_importances_.tolist()
        bar_data = [{"name": n, "value": round(v, 4)} for n, v in zip(FEATURE_NAMES, importances)]

        # 尝试读取 Data.py 生成的 correlation 数据
        result_path = os.path.join(BASE_PATH, "data_show", "result.json")
        correlation_matrix = []
        if os.path.exists(result_path):
            correlation_matrix = _compute_correlation(result_path)

        return {
            "code": 200,
            "data": {
                "feature_importance": bar_data,
                "correlation_matrix": correlation_matrix
            }
        }
    except Exception as e:
        return {"code": 500, "message": str(e)}


def _compute_correlation(result_path):
    """从清洗后的数据计算特征相关性矩阵"""
    import pandas as pd
    df = pd.read_json(result_path)
    numeric_df = df.select_dtypes(include=[np.number])
    cols = [c for c in FEATURE_NAMES if c in numeric_df.columns]
    if len(cols) < 2:
        return []
    corr = numeric_df[cols].corr().round(4)
    return corr.values.tolist()


@app.get("/api/tree")
def get_decision_tree():
    if clf_model is None:
        raise HTTPException(status_code=500, detail="模型未加载")
    if not hasattr(clf_model, "tree_"):
        return {"code": 500, "message": "当前模型不支持提取树结构"}

    def recurse(node):
        tree = clf_model.tree_
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            name = FEATURE_NAMES[tree.feature[node]] if tree.feature[node] < len(FEATURE_NAMES) else f"f{tree.feature[node]}"
            threshold = tree.threshold[node]
            return {
                "name": f"{name} <= {threshold:.2f}",
                "children": [
                    recurse(tree.children_left[node]),
                    recurse(tree.children_right[node])
                ]
            }
        else:
            # 叶子节点: value 是各类别计数/概率，取最大类
            val = tree.value[node][0]
            label_idx = int(np.argmax(val))
            label = CLASS_LABELS[label_idx] if label_idx < len(CLASS_LABELS) else str(label_idx)
            return {"name": f"{label}\n({val[label_idx]:.0f})"}

    try:
        echarts_tree_data = recurse(0)
        return {"code": 200, "data": echarts_tree_data}
    except Exception as e:
        return {"code": 500, "message": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)
