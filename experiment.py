from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from lightgbm_model import create_lightgbm_model, train_model, save_model
from toniot_dataset import ensure_processed_dataset
from xai import XAIConfig, run_xai_analysis


PROCESSED_DIR = Path("data_network/processed_type")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports/lightgbm")
XAI_DIR = REPORT_DIR / "xai"

RANDOM_STATE = 42

CLASS_NAMES = [
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


def load_processed_data(processed_dir: Path):
    ensure_processed_dataset(processed_dir)

    X_train = pd.read_csv(processed_dir / "X_train.csv")
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(processed_dir / "y_test.csv").squeeze("columns")

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

        results.append(
            {
                "class_id": class_id,
                "samples": len(X_class),
                "total_inference_time_sec": total_time,
                "avg_inference_time_ms": avg_time_ms,
            }
        )

    classwise_df = pd.DataFrame(results)

    print("\n========== Class-wise Inference Time ==========")
    print(classwise_df)

    return classwise_df


def evaluate_model(model, X_test, y_test):
    start_time = time.perf_counter()

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    end_time = time.perf_counter()

    total_inference_time = end_time - start_time
    avg_inference_time_ms = (total_inference_time / len(X_test)) * 1000
    throughput = len(X_test) / total_inference_time

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "total_inference_time_sec": total_inference_time,
        "avg_inference_time_ms": avg_inference_time_ms,
        "throughput_samples_per_sec": throughput,
    }

    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
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

    classwise_time_df = evaluate_classwise_inference_time(model, X_test, y_test)

    return metrics, report, cm, y_pred, y_prob, classwise_time_df


def save_reports(metrics, report, cm, y_test, y_pred, y_prob, report_dir: Path):
    report_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([metrics]).to_csv(report_dir / "metrics.csv", index=False)

    with open(
        report_dir / "classification_report.txt", "w", encoding="utf-8"
    ) as file:
        file.write(report)

    pd.DataFrame(cm).to_csv(report_dir / "confusion_matrix.csv", index=False)

    prob_df = pd.DataFrame(
        y_prob,
        columns=[f"prob_class_{i}" for i in range(y_prob.shape[1])],
    )
    pred_df = pd.DataFrame(
        {
            "y_true": y_test.reset_index(drop=True),
            "y_pred": y_pred,
        }
    )
    prediction_df = pd.concat([pred_df, prob_df], axis=1)
    prediction_df.to_csv(report_dir / "predictions.csv", index=False)

    print(f"[INFO] Reports saved to: {report_dir}")


def save_feature_importance(model, feature_names, report_dir: Path):
    booster = model.booster_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "split_importance": booster.feature_importance(importance_type="split"),
            "gain_importance": booster.feature_importance(importance_type="gain"),
        }
    ).sort_values(by="gain_importance", ascending=False)

    importance_df.to_csv(report_dir / "feature_importance.csv", index=False)

    print("\n========== Top 20 Feature Importance ==========")
    print(importance_df.head(20))
    print(
        f"[INFO] Feature importance saved to: "
        f"{report_dir / 'feature_importance.csv'}"
    )


def save_confusion_matrix_plot(cm, class_names, report_dir: Path):
    report_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(cm, cmap="Blues")

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    threshold = cm.max() / 2
    for row in range(len(class_names)):
        for column in range(len(class_names)):
            text_color = "white" if cm[row, column] > threshold else "black"
            ax.text(
                column,
                row,
                f"{cm[row, column]:,}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    output_path = report_dir / "confusion_matrix.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"[INFO] Confusion matrix plot saved to: {output_path}")


def run_xai_feature_selection(X_train, y_train):
    """Generate a leakage-safe feature list from a validation split.

    The final test set is not used for feature selection. A separate LightGBM
    model is fitted on X_fit and explained/evaluated on X_validation.
    """
    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    selection_model = create_lightgbm_model(
        num_classes=len(CLASS_NAMES),
        random_state=RANDOM_STATE,
    )
    selection_model = train_model(selection_model, X_fit, y_fit)

    config = XAIConfig(
        random_state=RANDOM_STATE,
        # Change these for each phase, e.g. 20 -> 10 -> 5.
        top_k_shap=10,
        top_k_cpf=10,
        min_features_per_class=3,
        cpf_metric="f1",
        cpf_repeats=5,
        cpf_n_conditioners=2,
        cpf_n_bins=5,
        shap_max_samples=3000,
        interaction_max_samples=1000,
        cpf_max_samples=None,
        enable_interaction=True,
        top_interaction_pairs_per_class=3,
        interaction_add_orphan_pairs=False,
        # LIME is optional and is excluded from automatic selection by default.
        enable_lime=False,
        include_lime_in_selection=False,
    )

    return run_xai_analysis(
        model=selection_model,
        X_reference=X_fit,
        y_reference=y_fit,
        X_eval=X_validation,
        y_eval=y_validation,
        class_names=CLASS_NAMES,
        output_dir=XAI_DIR,
        config=config,
    )


def run_lightgbm_pipeline(save_results=True, run_xai=True):
    X_train, X_test, y_train, y_test = load_processed_data(PROCESSED_DIR)

    # Final baseline model: train on all training rows and evaluate once on test.
    model = create_lightgbm_model(
        num_classes=len(CLASS_NAMES),
        random_state=RANDOM_STATE,
    )
    model = train_model(model, X_train, y_train)

    metrics, report, cm, y_pred, y_prob, classwise_time_df = evaluate_model(
        model,
        X_test,
        y_test,
    )

    if save_results:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        save_confusion_matrix_plot(cm, CLASS_NAMES, REPORT_DIR)
        classwise_time_df.to_csv(
            REPORT_DIR / "classwise_inference_time.csv",
            index=False,
        )
        save_reports(
            metrics,
            report,
            cm,
            y_test,
            y_pred,
            y_prob,
            REPORT_DIR,
        )
        save_feature_importance(model, X_train.columns, REPORT_DIR)
        save_model(model, MODEL_DIR)

        if run_xai:
            xai_result = run_xai_feature_selection(X_train, y_train)
            print("\n[INFO] Features for the next reduction phase:")
            print(xai_result.final_features)
    else:
        print("Terminate experiment without saving results")


if __name__ == "__main__":
    run_lightgbm_pipeline(save_results=True, run_xai=True)
