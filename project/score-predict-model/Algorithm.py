import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class Algorithm:
    def __init__(self, max_depth=5, random_state=42):
        self.clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        self.lr = Ridge(alpha=10.0)
        self.feature_names = None

    def fit(self, X_train, y_train_reg, y_train_clf):
        if hasattr(X_train, 'columns'):
            self.feature_names = X_train.columns.tolist()
        self.lr.fit(X_train, y_train_reg)
        self.clf.fit(X_train, y_train_clf)

    def predict(self, X_test):
        pred_reg = self.lr.predict(X_test)
        pred_clf = self.clf.predict(X_test)
        return {
            "regression_predictions": pred_reg,
            "classification_predictions": pred_clf
        }

    def LinearRegression_assessment(self, X_test, y_test_reg):
        y_pred = self.lr.predict(X_test)
        r2 = r2_score(y_test_reg, y_pred)
        mae = mean_absolute_error(y_test_reg, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test_reg, y_pred)))

        print("\n" + "=" * 30)
        print(" 模型 A (多元线性回归) 评估结果")
        print("=" * 30)
        print(f"R^2 (决定系数): {r2:.4f}")
        print(f"MAE (平均绝对误差): {mae:.4f}")
        print(f"RMSE (均方根误差): {rmse:.4f}")

        feature_weights = []
        if self.feature_names:
            print("\n--- 特征权重 (Weights) 分析 ---")
            for name, weight in zip(self.feature_names, self.lr.coef_):
                direction = "正相关" if weight > 0 else "负相关"
                print(f"{name}: {weight:.4f} ({direction})")
                feature_weights.append({"name": name, "weight": round(float(weight), 4), "direction": direction})
        else:
            print(f"权重数组: {self.lr.coef_}")

        return {"r2": round(float(r2), 4), "mae": round(float(mae), 4), "rmse": round(rmse, 4), "feature_weights": feature_weights}

    def tree_assessment(self, X_test, y_test_clf):
        y_pred = self.clf.predict(X_test)
        accuracy = accuracy_score(y_test_clf, y_pred)
        cm = confusion_matrix(y_test_clf, y_pred)
        precision = precision_score(y_test_clf, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test_clf, y_pred, average='weighted', zero_division=0)

        print("\n" + "=" * 30)
        print(" 模型 B (分类决策树) 评估结果")
        print("=" * 30)
        print(f"准确率 (Accuracy): {accuracy * 100:.2f}%")
        print(f"精确率 (Precision - weighted): {precision:.4f}")
        print(f"召回率 (Recall - weighted): {recall:.4f}")
        print("混淆矩阵 (Confusion Matrix):")
        print(cm)

        return {
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "confusion_matrix": cm.tolist()
        }
