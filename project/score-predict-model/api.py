import os
import json
import threading
import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.tree import _tree

app = FastAPI(title="Student Score Predict API")

BASE_PATH = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_PATH, "saved_models")

# 模型读写锁，防止训练和预测并发导致的数据不一致
_model_lock = threading.Lock()

lr_model = None
clf_model = None
metadata = {}

FEATURE_NAMES = []
CLASS_LABELS = ["不及格", "中", "良", "优"]
DISPLAY_FEATURES = []
_HAS_ONEHOT = False
_INTERACTION_IDX = 0


def reload_models():
    """重新加载保存的模型和元数据（调用方需持有 _model_lock）

    原子操作：先加载到临时变量，验证一致性后再更新全局状态。
    任何一步失败都不会修改全局状态。
    """
    global lr_model, clf_model, metadata, FEATURE_NAMES, CLASS_LABELS, DISPLAY_FEATURES, _HAS_ONEHOT, _INTERACTION_IDX

    lr_path = os.path.join(MODEL_DIR, "lr_model.pkl")
    clf_path = os.path.join(MODEL_DIR, "clf_model.pkl")
    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")

    for p in [lr_path, clf_path, meta_path]:
        if not os.path.exists(p):
            print(f"模型文件不存在: {p}，跳过加载")
            return

    # 先加载到临时变量
    try:
        _lr = joblib.load(lr_path)
        _clf = joblib.load(clf_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            _meta = json.load(f)
    except Exception as e:
        print(f"模型文件读取失败: {e}，保持当前模型不变")
        return

    _feature_names = _meta.get("feature_names", [])
    _class_labels = _meta.get("class_labels", ["不及格", "中", "良", "优"])

    # 一致性校验：元数据中的特征数必须和模型期望的输入维度一致
    expected = _lr.n_features_in_ if hasattr(_lr, 'n_features_in_') else len(_feature_names)
    if len(_feature_names) != expected:
        print(f"错误: 元数据特征数({len(_feature_names)})与模型期望维度({expected})不一致，保持当前模型不变")
        return

    # 全部校验通过，原子更新全局状态
    lr_model = _lr
    clf_model = _clf
    metadata = _meta
    FEATURE_NAMES = _feature_names
    CLASS_LABELS = _class_labels
    DISPLAY_FEATURES = [f for f in FEATURE_NAMES if "参与度等级" not in f]
    _HAS_ONEHOT = any("参与度等级" in f for f in FEATURE_NAMES)
    _INTERACTION_IDX = DISPLAY_FEATURES.index("线下_互动") if "线下_互动" in DISPLAY_FEATURES else 0
    print(f"模型加载成功，特征({len(FEATURE_NAMES)}): {FEATURE_NAMES}")


# 启动时加载
with _model_lock:
    reload_models()
if lr_model is None:
    print("模型加载失败，请先运行 main.py 训练模型")


def _build_features(raw_features: list[float]) -> np.ndarray:
    """将前端传入的原始特征扩展为模型期望的完整特征向量（含独热编码）"""
    X = np.array([raw_features], dtype=float)
    if not _HAS_ONEHOT:
        return X
    interact_val = raw_features[_INTERACTION_IDX]
    if interact_val <= 40:
        medium, high = 0, 0
    elif interact_val <= 70:
        medium, high = 1, 0
    else:
        medium, high = 0, 1
    onehot = np.array([[medium, high]], dtype=float)
    return np.concatenate([X, onehot], axis=1)


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/api/health")
def health():
    """健康检查"""
    with _model_lock:
        loaded = lr_model is not None and clf_model is not None
    return {
        "code": 200,
        "data": {
            "status": "ok" if loaded else "no_model",
            "model_loaded": loaded,
            "feature_count": len(FEATURE_NAMES),
            "features": FEATURE_NAMES,
        }
    }


@app.get("/api/features")
def get_features():
    with _model_lock:
        if lr_model is None or clf_model is None:
            raise HTTPException(status_code=500, detail="模型未加载，请先运行 main.py 训练模型")
        return {
            "code": 200,
            "data": {
                "feature_names": list(DISPLAY_FEATURES),
                "class_labels": list(CLASS_LABELS)
            }
        }


@app.post("/api/predict")
def predict_score(req: PredictRequest):
    with _model_lock:
        if lr_model is None or clf_model is None:
            raise HTTPException(status_code=500, detail="模型未加载，请先运行 main.py 训练模型")
        if len(req.features) != len(DISPLAY_FEATURES):
            raise HTTPException(
                status_code=400,
                detail=f"特征数量不匹配，期望 {len(DISPLAY_FEATURES)} 个 ({DISPLAY_FEATURES})，实际收到 {len(req.features)} 个。请刷新页面。"
            )

        try:
            X = _build_features(req.features)
            expected_dim = lr_model.n_features_in_ if hasattr(lr_model, 'n_features_in_') else len(FEATURE_NAMES)
            if X.shape[1] != expected_dim:
                raise HTTPException(
                    status_code=400,
                    detail=f"模型期望 {expected_dim} 维特征，但构造出 {X.shape[1]} 维。请刷新页面获取最新特征列表。"
                )

            score_reg = float(np.clip(lr_model.predict(X)[0], 0.0, 100.0))
            label_clf = str(clf_model.predict(X)[0])
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

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
    with _model_lock:
        if clf_model is None:
            raise HTTPException(status_code=500, detail="模型未加载")

        try:
            importances = clf_model.feature_importances_.tolist()
            bar_data = [{"name": n, "value": round(v, 4)} for n, v in zip(FEATURE_NAMES, importances)]
        except Exception as e:
            return {"code": 500, "message": f"读取特征重要性失败: {str(e)}"}

    # 相关性矩阵不依赖模型锁
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


def _compute_correlation(result_path):
    import pandas as pd
    try:
        df = pd.read_json(result_path)
        numeric_df = df.select_dtypes(include=[np.number])
        cols = [c for c in FEATURE_NAMES if c in numeric_df.columns]
        if len(cols) < 2:
            return []
        corr = numeric_df[cols].corr().round(4)
        return corr.values.tolist()
    except Exception as e:
        print(f"计算相关性矩阵失败: {e}")
        return []


@app.get("/api/tree")
def get_decision_tree():
    with _model_lock:
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
                val = tree.value[node][0]
                label_idx = int(np.argmax(val))
                label = CLASS_LABELS[label_idx] if label_idx < len(CLASS_LABELS) else str(label_idx)
                return {"name": f"{label}\n({val[label_idx]:.0f})"}

        try:
            echarts_tree_data = recurse(0)
            return {"code": 200, "data": echarts_tree_data}
        except Exception as e:
            return {"code": 500, "message": str(e)}


# ==================== 模型训练接口 ====================

class TrainConfig(BaseModel):
    target_col: str = "线上总成绩"
    test_size: float = 0.2
    random_state: int = 42
    max_depth: int = 5
    correlation_threshold: float = 0.1
    feature_names: list[str] | None = None
    class_bins: list[float] = [-1.0, 59.9, 79.9, 89.9, 101.0]
    class_labels: list[str] = ["不及格", "中", "良", "优"]
    range_limit: dict | None = None
    data_source: str = "excel"


@app.get("/api/train/options")
def get_train_options(source: str = "excel"):
    try:
        if source == "db":
            from main import get_available_features_from_db
            info = get_available_features_from_db()
        else:
            from main import get_available_features
            info = get_available_features()
        return {"code": 200, "data": info}
    except Exception as e:
        return {"code": 500, "message": f"读取数据失败: {str(e)}"}


@app.post("/api/train")
def train(config: TrainConfig):
    try:
        if config.data_source == "db":
            from main import train_model_from_db
            train_fn = train_model_from_db
        else:
            from main import train_model
            train_fn = train_model

        cfg = {
            "target_col": config.target_col,
            "test_size": config.test_size,
            "random_state": config.random_state,
            "max_depth": config.max_depth,
            "correlation_threshold": config.correlation_threshold,
            "feature_names": config.feature_names,
            "class_bins": config.class_bins,
            "class_labels": config.class_labels,
        }
        if config.range_limit is not None:
            cfg["range_limit"] = config.range_limit

        result = train_fn(cfg)

        # 训练完成后加锁加载新模型
        with _model_lock:
            reload_models()

        return {
            "code": 200,
            "message": "模型训练完成",
            "data": {
                "feature_names": result["feature_names"],
                "target_col": result["target_col"],
                "class_labels": result["class_labels"],
                "metrics": result["metrics"],
                "feature_importance": result["feature_importance"],
                "correlation_matrix": result["correlation_matrix"],
                "correlation_labels": result["correlation_labels"],
                "config": result["config"]
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"code": 500, "message": f"训练失败: {str(e)}"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)
