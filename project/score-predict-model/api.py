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

# 前端展示用的原始特征名（5个数值型，不含独热编码列）
DISPLAY_FEATURES = [f for f in FEATURE_NAMES if "参与度等级" not in f]

# 预计算索引，用于在预测时自动补充独热编码特征
_HAS_ONEHOT = any("参与度等级" in f for f in FEATURE_NAMES)
_INTERACTION_IDX = DISPLAY_FEATURES.index("线下_互动") if "线下_互动" in DISPLAY_FEATURES else 0


def _build_features(raw_features: list[float]) -> np.ndarray:
    """将前端传入的原始 5 个特征扩展为模型期望的完整特征向量（含独热编码）"""
    X = np.array([raw_features], dtype=float)
    if not _HAS_ONEHOT:
        return X
    # 根据 线下_互动 值推导参与度等级并独热编码
    interact_val = raw_features[_INTERACTION_IDX]
    if interact_val <= 40:
        medium, high = 0, 0  # 低参与度（参考类别）
    elif interact_val <= 70:
        medium, high = 1, 0  # 中参与度
    else:
        medium, high = 0, 1  # 高参与度
    onehot = np.array([[medium, high]], dtype=float)
    return np.concatenate([X, onehot], axis=1)


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/api/features")
def get_features():
    """返回前端可用的特征名称（原始数值特征，不含独热编码列）"""
    if not FEATURE_NAMES:
        raise HTTPException(status_code=500, detail="模型未加载")
    return {
        "code": 200,
        "data": {
            "feature_names": DISPLAY_FEATURES,
            "class_labels": CLASS_LABELS
        }
    }


@app.post("/api/predict")
def predict_score(req: PredictRequest):
    if lr_model is None or clf_model is None:
        raise HTTPException(status_code=500, detail="模型未加载，请先运行 main.py 训练模型")
    if len(req.features) != len(DISPLAY_FEATURES):
        raise HTTPException(
            status_code=400,
            detail=f"特征数量不匹配，期望 {len(DISPLAY_FEATURES)} 个: {DISPLAY_FEATURES}"
        )

    X = _build_features(req.features)

    # 线性回归预测连续分数，裁剪至 [0, 100] 区间
    score_reg = float(np.clip(lr_model.predict(X)[0], 0.0, 100.0))
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
