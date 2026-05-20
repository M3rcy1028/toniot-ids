from pathlib import Path
import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import matplotlib.pyplot as plt
import numpy as np

PROCESSED_DIR = Path("data/processed_type") # classification
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports/lightgbm")

RANDOM_STATE = 42

def load_processed_data(processed_dir: Path):
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv").squeeze()
    y_test = pd.read_csv(processed_dir / "y_test.csv").squeeze()

    print("[INFO] Processed data loaded")
    print(f"[INFO] X_train: {X_train.shape}")
    print(f"[INFO] X_test : {X_test.shape}")

    return X_train, X_test, y_train, y_test


def create_lightgbm_model(num_classes: int) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        boosting_type="gbdt",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    return model


def train_model(model, X_train, y_train):
    print("[INFO] Training LightGBM model...")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train)],
        eval_metric="binary_logloss",
    )

    print("[INFO] Training complete")
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(
            y_test,
            y_pred,
            average="macro"
        ),
        "recall_macro": recall_score(
            y_test,
            y_pred,
            average="macro"
        ),
        "f1_macro": f1_score(
            y_test,
            y_pred,
            average="macro"
        ),
    }

    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n========== Evaluation Results ==========")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    print("\n========== Classification Report ==========")
    print(report)

    print("\n========== Confusion Matrix ==========")
    print(cm)

    return metrics, report, cm, y_pred, y_prob


def save_model(model, model_dir: Path):
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "lightgbm_toniot_classification.pkl"
    joblib.dump(model, model_path)

    print(f"[INFO] Model saved to: {model_path}")


def save_reports(metrics, report, cm, y_pred, y_prob, report_dir: Path):
    report_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([metrics]).to_csv(report_dir / "metrics.csv", index=False)

    with open(report_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    pd.DataFrame(cm).to_csv(report_dir / "confusion_matrix.csv", index=False)

    pd.DataFrame({
        "y_pred": y_pred,
        "y_prob": y_prob,
    }).to_csv(report_dir / "predictions.csv", index=False)

    print(f"[INFO] Reports saved to: {report_dir}")


def save_feature_importance(model, feature_names, report_dir: Path):
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values(by="importance", ascending=False)

    importance_df.to_csv(report_dir / "feature_importance.csv", index=False)

    print("\n========== Top 20 Feature Importance ==========")
    print(importance_df.head(20))

    print(f"[INFO] Feature importance saved to: {report_dir / 'feature_importance.csv'}")

def save_confusion_matrix_plot(cm, class_names, report_dir: Path):
    report_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    # white → blue colormap
    im = ax.imshow(cm, cmap="Blues")

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))

    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    threshold = cm.max() / 2

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text_color = "white" if cm[i, j] > threshold else "black"

            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    output_path = report_dir / "confusion_matrix.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"[INFO] Confusion matrix plot saved to: {output_path}")

def run_lightgbm_pipeline():
    X_train, X_test, y_train, y_test = load_processed_data(PROCESSED_DIR)

    model = create_lightgbm_model(10)
    model = train_model(model, X_train, y_train)

    metrics, report, cm, y_pred, y_prob = evaluate_model(model, X_test, y_test)

    class_names = [
        "backdoor",
        "ddos",
        "dos",
        "injection",
        "mitm",
        "normal",
        "password",
        "ransomware",
        "scanning",
        "xss",
    ]

    save_confusion_matrix_plot(cm, class_names, REPORT_DIR)

    save_model(model, MODEL_DIR)
    save_reports(metrics, report, cm, y_pred, y_prob, REPORT_DIR)
    save_feature_importance(model, X_train.columns, REPORT_DIR)