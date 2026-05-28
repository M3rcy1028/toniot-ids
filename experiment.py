from pathlib import Path
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from lightgbm_model import create_lightgbm_model, train_model, save_model


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


def evaluate_classwise_inference_time(model, X_test, y_test):
    results = []

    for class_id in sorted(y_test.unique()):
        class_indices = y_test[y_test == class_id].index
        X_class = X_test.loc[class_indices]

        start_time = time.perf_counter()
        _ = model.predict(X_class)
        end_time = time.perf_counter()

        total_time = end_time - start_time
        avg_time_ms = (total_time / len(X_class)) * 1000

        results.append({
            "class_id": class_id,
            "samples": len(X_class),
            "total_inference_time_sec": total_time,
            "avg_inference_time_ms": avg_time_ms,
        })

    classwise_df = pd.DataFrame(results)

    print("\n========== Class-wise Inference Time ==========")
    print(classwise_df)

    return classwise_df

def evaluate_model(model, X_test, y_test):
    # Batch Inference (normal + anomalous)
    start_time = time.perf_counter()

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    end_time = time.perf_counter()

    total_inference_time = end_time - start_time
    avg_inference_time_ms = (total_inference_time / len(X_test)) * 1000
    throughput = len(X_test) / total_inference_time

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro"),
        "recall_macro": recall_score(y_test, y_pred, average="macro"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "total_inference_time_sec": total_inference_time,
        "avg_inference_time_ms": avg_inference_time_ms,
        "throughput_samples_per_sec": throughput,
    }

    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    print("\n========== Evaluation Results ==========")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    print("\n========== Classification Report ==========")
    print(report)

    print("\n========== Confusion Matrix ==========")
    print(cm)

    print("\n========== Batch Inference Time ==========")
    print(f"Total inference time: {total_inference_time:.6f} sec")
    print(f"Average inference time: {avg_inference_time_ms:.6f} ms/sample")
    print(f"Throughput: {throughput:.2f} samples/sec")

    # Class-wise Inference
    classwise_time_df = evaluate_classwise_inference_time(model, X_test, y_test)

    return metrics, report, cm, y_pred, y_prob, classwise_time_df


def save_reports(metrics, report, cm, y_test, y_pred, y_prob, report_dir: Path):
    report_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([metrics]).to_csv(report_dir / "metrics.csv", index=False)

    with open(report_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    pd.DataFrame(cm).to_csv(report_dir / "confusion_matrix.csv", index=False)

    # multiclass probability 저장
    prob_df = pd.DataFrame(
        y_prob,
        columns=[f"prob_class_{i}" for i in range(y_prob.shape[1])]
    )

    pred_df = pd.DataFrame({
        "y_true": y_test.reset_index(drop=True),
        "y_pred": y_pred
    })

    prediction_df = pd.concat([pred_df, prob_df], axis=1)
    prediction_df.to_csv(report_dir / "predictions.csv", index=False)

    print(f"[INFO] Reports saved to: {report_dir}")


def save_feature_importance(model, feature_names, report_dir: Path):
    booster = model.booster_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "split_importance": booster.feature_importance(importance_type="split"),
        "gain_importance": booster.feature_importance(importance_type="gain"),
    })

    importance_df = importance_df.sort_values(
        by="gain_importance",
        ascending=False
    )

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

def run_lightgbm_pipeline(save_results=True):
    X_train, X_test, y_train, y_test = load_processed_data(PROCESSED_DIR)

    model = create_lightgbm_model(num_classes=10, random_state=RANDOM_STATE)
    model = train_model(model, X_train, y_train)

    metrics, report, cm, y_pred, y_prob, classwise_time_df = evaluate_model(model, X_test, y_test)

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

    if (save_results):
        save_confusion_matrix_plot(cm, class_names, REPORT_DIR)
        classwise_time_df.to_csv(REPORT_DIR / "classwise_inference_time.csv", index=False)
        save_reports(metrics, report, cm, y_test, y_pred, y_prob, REPORT_DIR)
        save_feature_importance(model, X_train.columns, REPORT_DIR)

        save_model(model, MODEL_DIR)
    else:
        print("Terminate experiment without saving results")