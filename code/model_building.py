from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


def load_features(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def get_feature_matrix(df: pd.DataFrame):

    feature_columns = [col for col in df.columns if col not in ("account.id", "label")]
    X = df[feature_columns]
    y = df["label"]
    return X, y, feature_columns


def evaluate_model(model, model_name: str, X: pd.DataFrame, y: pd.Series, folds: KFold):
    scores = []
    for fold_idx, (train_index, val_index) in enumerate(folds.split(X), start=1):
        X_train_fold = X.iloc[train_index]
        X_val_fold = X.iloc[val_index]
        y_train_fold = y.iloc[train_index]
        y_val_fold = y.iloc[val_index]

        model.fit(X_train_fold, y_train_fold)
        preds = model.predict(X_val_fold)
        fold_accuracy = accuracy_score(y_val_fold, preds)
        scores.append(fold_accuracy)
        print(f"[{model_name}] Fold {fold_idx} accuracy: {fold_accuracy:.4f}")

    mean_score = float(np.mean(scores)) if scores else float("nan")
    print(f"[{model_name}] Mean cross-validation accuracy: {mean_score:.4f}\n")
    return scores


def main():
    features = load_features(Path("model_ready_features.csv"))
    X, y, _ = get_feature_matrix(features)

    folds = KFold(n_splits=5, shuffle=True, random_state=42)

    logistic_model = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        random_state=42,
    )
    evaluate_model(logistic_model, "LogisticRegression", X, y, folds)

    xgb_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
        use_label_encoder=False,
    )
    evaluate_model(xgb_model, "XGBClassifier", X, y, folds)


if __name__ == "__main__":
    main()
