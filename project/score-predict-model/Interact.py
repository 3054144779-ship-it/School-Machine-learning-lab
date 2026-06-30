"""交互层：命令行成绩预测工具"""
import os
import json
import joblib
import numpy as np


class Interact:
    def __init__(self):
        base_path = os.path.dirname(__file__)
        model_dir = os.path.join(base_path, "saved_models")

        self.lr_model = joblib.load(os.path.join(model_dir, "lr_model.pkl"))
        self.clf_model = joblib.load(os.path.join(model_dir, "clf_model.pkl"))

        with open(os.path.join(model_dir, "model_metadata.json"), "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.all_features = metadata["feature_names"]
        self.class_labels = metadata["class_labels"]

        # 前端交互用原始数值特征（去掉独热编码列）
        self.feature_names = [f for f in self.all_features if "参与度等级" not in f]
        self._has_onehot = any("参与度等级" in f for f in self.all_features)

    def _build_features(self, raw: list[float]) -> np.ndarray:
        X = np.array([raw], dtype=float)
        if not self._has_onehot:
            return X
        interact_val = raw[0]  # 线下_互动 是第一个特征
        if interact_val <= 40:
            medium, high = 0, 0
        elif interact_val <= 70:
            medium, high = 1, 0
        else:
            medium, high = 0, 1
        return np.concatenate([X, np.array([[medium, high]], dtype=float)], axis=1)

    def predict(self, features):
        X = self._build_features(features)
        score = round(float(np.clip(self.lr_model.predict(X)[0], 0.0, 100.0)), 2)
        label = str(self.clf_model.predict(X)[0])
        return score, label

    def run(self):
        print("=" * 50)
        print("  学生成绩预测系统 — 交互预测")
        print("=" * 50)
        print(f"\n模型特征 ({len(self.feature_names)} 个):")
        for i, name in enumerate(self.feature_names, 1):
            print(f"  {i}. {name}")
        if self._has_onehot:
            print("\n（注：'参与度等级' 类别特征会根据 '线下_互动' 值自动派生）")

        while True:
            print("\n" + "-" * 40)
            print("请输入学生各项特征值 (输入 q 退出):")

            features = []
            for name in self.feature_names:
                while True:
                    val = input(f"  {name} (0-100): ").strip()
                    if val.lower() == 'q':
                        print("已退出。")
                        return
                    try:
                        v = float(val)
                        if 0 <= v <= 100:
                            features.append(v)
                            break
                        else:
                            print("    请输入 0-100 之间的数值")
                    except ValueError:
                        print("    请输入有效数字")

            score, label = self.predict(features)

            print(f"\n  >>> 预测结果 <<<")
            print(f"  预测分数: {score} 分")
            print(f"  预测等级: {label}")


if __name__ == "__main__":
    try:
        interact = Interact()
        interact.run()
    except FileNotFoundError as e:
        print(f"模型文件未找到，请先运行 main.py 训练模型: {e}")
    except KeyboardInterrupt:
        print("\n\n已退出。")
