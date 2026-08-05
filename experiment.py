from pathlib import Path
from datetime import datetime
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
from lightgbm_model import create_lightgbm_model, train_model, save_model
from toniot_dataset import ensure_processed_dataset
from xai import XAIConfig, run_xai_analysis


PROCESSED_DIR = Path("data_network/processed_type")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports") / f"lightgbm_{datetime.now():%y%m%d}"
XAI_DIR = REPORT_DIR / "xai"
REDUCTION_1_DIR = REPORT_DIR / "reduction_1"
REDUCTION_1_XAI_DIR = XAI_DIR / "reduction_1"
REDUCTION_2_DIR = REPORT_DIR / "reduction_2"
REDUCTION_2_XAI_DIR = XAI_DIR / "reduction_2"
REDUCTION_3_DIR = REPORT_DIR / "reduction_3"
REDUCTION_3_XAI_DIR = XAI_DIR / "reduction_3"
REDUCTION_1_FEATURES = (
    REDUCTION_1_XAI_DIR / "selection" / "final_selected_features.csv"
)
REDUCTION_2_FEATURES = (
    REDUCTION_2_XAI_DIR / "selection" / "final_selected_features.csv"
)

RANDOM_STATE = 42

CLASS_NAMES = [
    "backdoor",     # 0
    "ddos",         # 1
    "dos",          # 2
    "injection",    # 3
    "mitm",         # 4
    "normal",       # 5
    "password",     # 6
    "ransomware",   # 7
    "scanning",     # 8
    "xss",          # 9
]


def load_processed_data(processed_dir: Path, feature_selection=False, fs_config=None):
    ensure_processed_dataset(processed_dir)

    X_train = pd.read_csv(processed_dir / "X_train.csv")
    X_valid = pd.read_csv(processed_dir / "X_valid.csv")
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv").squeeze("columns")
    y_valid = pd.read_csv(processed_dir / "y_valid.csv").squeeze("columns")
    y_test = pd.read_csv(processed_dir / "y_test.csv").squeeze("columns")

    if feature_selection:
        print("[INFO] Feature selection enabled")
        selected_features = pd.read_csv(fs_config)["feature"].tolist()

        X_train = X_train[selected_features]
        X_valid = X_valid[selected_features]
        X_test = X_test[selected_features]

    print("[INFO] Processed data loaded")
    print(f"[INFO] X_train: {X_train.shape}")
    print(f"[INFO] X_valid: {X_valid.shape}")
    print(f"[INFO] X_test : {X_test.shape}")
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test


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


def run_xai_feature_selection(
    X_train,
    y_train,
    X_valid,
    y_valid,
    output_dir=XAI_DIR,
    config=None,
):
    """Generate a leakage-safe feature list with the validation split.

    The final test set is not used for feature selection. A separate LightGBM
    model is fitted on the training set and explained/evaluated on validation.
    """
    selection_model = create_lightgbm_model(
        num_classes=len(CLASS_NAMES),
        random_state=RANDOM_STATE,
    )
    selection_model = train_model(
        selection_model,
        X_train,
        y_train,
        X_valid,
        y_valid,
    )

    if config is None:
        config = XAIConfig(
            random_state=RANDOM_STATE,
            top_k_sage=14,
            top_k_shap=12,
            top_k_cpf=12,
            min_features_per_class=3,
            min_final_features=18,
            max_final_features=20,
            cpf_metric="f1",
            cpf_repeats=5,
            cpf_n_conditioners=2,
            cpf_n_bins=5,
            shap_batch_size=4096,
            interaction_batch_size=256,
            sage_repeats=1,
            cpf_max_samples=None,
            correlation_max_samples=None,
            enable_interaction=True,
            top_interaction_pairs_per_class=3,
            interaction_add_orphan_pairs=False,
            # LIME is optional and excluded from automatic selection by default.
            enable_lime=False,
            include_lime_in_selection=False,
        )

    return run_xai_analysis(
        model=selection_model,
        X_reference=X_train,
        y_reference=y_train,
        X_eval=X_valid,
        y_eval=y_valid,
        class_names=CLASS_NAMES,
        output_dir=output_dir,
        config=config,
    )


def _selection_metrics(model, X, y):
    y_pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision_macro": precision_score(
            y, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y, y_pred, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
        "mitm_recall": recall_score(
            y == 4,
            y_pred == 4,
            zero_division=0,
        ),
    }


def run_reduction_1_experiment():
    """Run full-train XAI and evaluate the first 18-20 feature reduction."""
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_processed_data(
        PROCESSED_DIR
    )

    baseline_model = create_lightgbm_model(
        num_classes=len(CLASS_NAMES),
        random_state=RANDOM_STATE,
    )
    baseline_model = train_model(
        baseline_model,
        X_train,
        y_train,
        X_valid,
        y_valid,
    )
    baseline_validation = _selection_metrics(
        baseline_model,
        X_valid,
        y_valid,
    )

    xai_result = run_xai_feature_selection(
        X_train,
        y_train,
        X_valid,
        y_valid,
        output_dir=REDUCTION_1_XAI_DIR,
    )
    selected_features = xai_result.final_features
    print(f"[INFO] Reduction-1 selected features ({len(selected_features)}):")
    print(selected_features)

    reduced_model = create_lightgbm_model(
        num_classes=len(CLASS_NAMES),
        random_state=RANDOM_STATE,
    )
    reduced_model = train_model(
        reduced_model,
        X_train[selected_features],
        y_train,
        X_valid[selected_features],
        y_valid,
    )
    reduced_validation = _selection_metrics(
        reduced_model,
        X_valid[selected_features],
        y_valid,
    )

    REDUCTION_1_DIR.mkdir(parents=True, exist_ok=True)
    validation_comparison = pd.DataFrame(
        [
            {"model": "baseline_22", **baseline_validation},
            {
                "model": f"reduction_1_{len(selected_features)}",
                **reduced_validation,
            },
        ]
    )
    validation_comparison.to_csv(
        REDUCTION_1_DIR / "validation_comparison.csv",
        index=False,
    )
    print("\n========== Validation Comparison ==========")
    print(validation_comparison.to_string(index=False))

    accuracy_drop = baseline_validation["accuracy"] - reduced_validation["accuracy"]
    macro_f1_drop = baseline_validation["f1_macro"] - reduced_validation["f1_macro"]
    mitm_recall_drop = (
        baseline_validation["mitm_recall"] - reduced_validation["mitm_recall"]
    )
    accepted = (
        18 <= len(selected_features) <= 20
        and accuracy_drop <= 0.001
        and macro_f1_drop <= 0.002
        and mitm_recall_drop <= 0.01
    )
    decision = pd.DataFrame(
        [
            {
                "accepted": accepted,
                "selected_feature_count": len(selected_features),
                "accuracy_drop": accuracy_drop,
                "macro_f1_drop": macro_f1_drop,
                "mitm_recall_drop": mitm_recall_drop,
            }
        ]
    )
    decision.to_csv(REDUCTION_1_DIR / "acceptance_decision.csv", index=False)
    print("\n========== Reduction-1 Acceptance ==========")
    print(decision.to_string(index=False))

    if not accepted:
        print("[WARN] Reduction-1 rejected; test data was not evaluated.")
        return {
            "accepted": False,
            "selected_features": selected_features,
            "validation_comparison": validation_comparison,
        }

    metrics, report, cm, y_pred, y_prob, classwise_time_df = evaluate_model(
        reduced_model,
        X_test[selected_features],
        y_test,
    )
    save_confusion_matrix_plot(cm, CLASS_NAMES, REDUCTION_1_DIR)
    classwise_time_df.to_csv(
        REDUCTION_1_DIR / "classwise_inference_time.csv",
        index=False,
    )
    save_reports(
        metrics,
        report,
        cm,
        y_test,
        y_pred,
        y_prob,
        REDUCTION_1_DIR,
    )
    save_feature_importance(
        reduced_model,
        selected_features,
        REDUCTION_1_DIR,
    )
    save_model(
        reduced_model,
        MODEL_DIR,
        filename="lightgbm_toniot_classification_reduction_1.pkl",
    )
    return {
        "accepted": True,
        "selected_features": selected_features,
        "validation_comparison": validation_comparison,
        "test_metrics": metrics,
    }


def run_reduction_2_experiment():
    """Run full-train XAI on reduction-1 features and target 16-17 features."""
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_processed_data(
        PROCESSED_DIR
    )
    if not REDUCTION_1_FEATURES.exists():
        raise FileNotFoundError(
            f"Reduction-1 feature list not found: {REDUCTION_1_FEATURES}"
        )
    input_features = pd.read_csv(REDUCTION_1_FEATURES)["feature"].tolist()
    if len(input_features) != 18 or len(set(input_features)) != 18:
        raise ValueError(
            "Reduction-2 requires the verified 18 unique reduction-1 features"
        )

    X_train_input = X_train[input_features]
    X_valid_input = X_valid[input_features]
    baseline_model = create_lightgbm_model(
        num_classes=len(CLASS_NAMES),
        random_state=RANDOM_STATE,
    )
    baseline_model = train_model(
        baseline_model,
        X_train_input,
        y_train,
        X_valid_input,
        y_valid,
    )
    baseline_validation = _selection_metrics(
        baseline_model,
        X_valid_input,
        y_valid,
    )

    reduction_2_config = XAIConfig(
        random_state=RANDOM_STATE,
        top_k_sage=12,
        top_k_shap=10,
        top_k_cpf=10,
        min_features_per_class=3,
        min_final_features=16,
        max_final_features=17,
        cpf_metric="f1",
        cpf_repeats=5,
        cpf_n_conditioners=2,
        cpf_n_bins=5,
        shap_batch_size=4096,
        interaction_batch_size=256,
        sage_repeats=1,
        cpf_max_samples=None,
        correlation_max_samples=None,
        enable_interaction=True,
        top_interaction_pairs_per_class=3,
        interaction_add_orphan_pairs=False,
        enable_lime=False,
        include_lime_in_selection=False,
    )
    xai_result = run_xai_feature_selection(
        X_train_input,
        y_train,
        X_valid_input,
        y_valid,
        output_dir=REDUCTION_2_XAI_DIR,
        config=reduction_2_config,
    )
    selected_features = xai_result.final_features
    print(f"[INFO] Reduction-2 selected features ({len(selected_features)}):")
    print(selected_features)

    reduced_model = create_lightgbm_model(
        num_classes=len(CLASS_NAMES),
        random_state=RANDOM_STATE,
    )
    reduced_model = train_model(
        reduced_model,
        X_train[selected_features],
        y_train,
        X_valid[selected_features],
        y_valid,
    )
    reduced_validation = _selection_metrics(
        reduced_model,
        X_valid[selected_features],
        y_valid,
    )

    REDUCTION_2_DIR.mkdir(parents=True, exist_ok=True)
    validation_comparison = pd.DataFrame(
        [
            {"model": "reduction_1_18", **baseline_validation},
            {
                "model": f"reduction_2_{len(selected_features)}",
                **reduced_validation,
            },
        ]
    )
    validation_comparison.to_csv(
        REDUCTION_2_DIR / "validation_comparison.csv",
        index=False,
    )
    print("\n========== Validation Comparison ==========")
    print(validation_comparison.to_string(index=False))

    accuracy_drop = baseline_validation["accuracy"] - reduced_validation["accuracy"]
    macro_f1_drop = baseline_validation["f1_macro"] - reduced_validation["f1_macro"]
    mitm_recall_drop = (
        baseline_validation["mitm_recall"] - reduced_validation["mitm_recall"]
    )
    accepted = (
        16 <= len(selected_features) <= 17
        and accuracy_drop <= 0.001
        and macro_f1_drop <= 0.002
        and mitm_recall_drop <= 0.01
    )
    decision = pd.DataFrame(
        [
            {
                "accepted": accepted,
                "selected_feature_count": len(selected_features),
                "accuracy_drop": accuracy_drop,
                "macro_f1_drop": macro_f1_drop,
                "mitm_recall_drop": mitm_recall_drop,
            }
        ]
    )
    decision.to_csv(REDUCTION_2_DIR / "acceptance_decision.csv", index=False)
    print("\n========== Reduction-2 Acceptance ==========")
    print(decision.to_string(index=False))

    if not accepted:
        print("[WARN] Reduction-2 rejected; test data was not evaluated.")
        return {
            "accepted": False,
            "selected_features": selected_features,
            "validation_comparison": validation_comparison,
        }

    metrics, report, cm, y_pred, y_prob, classwise_time_df = evaluate_model(
        reduced_model,
        X_test[selected_features],
        y_test,
    )
    save_confusion_matrix_plot(cm, CLASS_NAMES, REDUCTION_2_DIR)
    classwise_time_df.to_csv(
        REDUCTION_2_DIR / "classwise_inference_time.csv",
        index=False,
    )
    save_reports(
        metrics,
        report,
        cm,
        y_test,
        y_pred,
        y_prob,
        REDUCTION_2_DIR,
    )
    save_feature_importance(
        reduced_model,
        selected_features,
        REDUCTION_2_DIR,
    )
    save_model(
        reduced_model,
        MODEL_DIR,
        filename="lightgbm_toniot_classification_reduction_2.pkl",
    )
    return {
        "accepted": True,
        "selected_features": selected_features,
        "validation_comparison": validation_comparison,
        "test_metrics": metrics,
    }


def run_reduction_3_experiment():
    """Run full-train XAI on reduction-2 features and target 14-15 features."""
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_processed_data(
        PROCESSED_DIR
    )
    if not REDUCTION_2_FEATURES.exists():
        raise FileNotFoundError(
            f"Reduction-2 feature list not found: {REDUCTION_2_FEATURES}"
        )
    reduction_2_decision_path = REDUCTION_2_DIR / "acceptance_decision.csv"
    if not reduction_2_decision_path.exists():
        raise FileNotFoundError(
            f"Reduction-2 acceptance decision not found: {reduction_2_decision_path}"
        )
    reduction_2_decision = pd.read_csv(reduction_2_decision_path)
    if reduction_2_decision.empty or not bool(
        reduction_2_decision.loc[0, "accepted"]
    ):
        raise ValueError("Reduction-3 requires an accepted reduction-2 result")

    input_features = pd.read_csv(REDUCTION_2_FEATURES)["feature"].tolist()
    if len(input_features) != 16 or len(set(input_features)) != 16:
        raise ValueError(
            "Reduction-3 requires the verified 16 unique reduction-2 features"
        )

    X_train_input = X_train[input_features]
    X_valid_input = X_valid[input_features]
    baseline_model = create_lightgbm_model(
        num_classes=len(CLASS_NAMES),
        random_state=RANDOM_STATE,
    )
    baseline_model = train_model(
        baseline_model,
        X_train_input,
        y_train,
        X_valid_input,
        y_valid,
    )
    baseline_validation = _selection_metrics(
        baseline_model,
        X_valid_input,
        y_valid,
    )

    reduction_3_config = XAIConfig(
        random_state=RANDOM_STATE,
        top_k_sage=10,
        top_k_shap=8,
        top_k_cpf=8,
        min_features_per_class=3,
        min_final_features=14,
        max_final_features=15,
        cpf_metric="f1",
        cpf_repeats=5,
        cpf_n_conditioners=2,
        cpf_n_bins=5,
        shap_batch_size=4096,
        interaction_batch_size=256,
        sage_repeats=1,
        cpf_max_samples=None,
        correlation_max_samples=None,
        enable_interaction=True,
        top_interaction_pairs_per_class=3,
        interaction_add_orphan_pairs=False,
        enable_lime=False,
        include_lime_in_selection=False,
    )
    xai_result = run_xai_feature_selection(
        X_train_input,
        y_train,
        X_valid_input,
        y_valid,
        output_dir=REDUCTION_3_XAI_DIR,
        config=reduction_3_config,
    )
    selected_features = xai_result.final_features
    print(f"[INFO] Reduction-3 selected features ({len(selected_features)}):")
    print(selected_features)

    reduced_model = create_lightgbm_model(
        num_classes=len(CLASS_NAMES),
        random_state=RANDOM_STATE,
    )
    reduced_model = train_model(
        reduced_model,
        X_train[selected_features],
        y_train,
        X_valid[selected_features],
        y_valid,
    )
    reduced_validation = _selection_metrics(
        reduced_model,
        X_valid[selected_features],
        y_valid,
    )

    REDUCTION_3_DIR.mkdir(parents=True, exist_ok=True)
    validation_comparison = pd.DataFrame(
        [
            {"model": "reduction_2_16", **baseline_validation},
            {
                "model": f"reduction_3_{len(selected_features)}",
                **reduced_validation,
            },
        ]
    )
    validation_comparison.to_csv(
        REDUCTION_3_DIR / "validation_comparison.csv",
        index=False,
    )
    print("\n========== Validation Comparison ==========")
    print(validation_comparison.to_string(index=False))

    accuracy_drop = baseline_validation["accuracy"] - reduced_validation["accuracy"]
    macro_f1_drop = baseline_validation["f1_macro"] - reduced_validation["f1_macro"]
    mitm_recall_drop = (
        baseline_validation["mitm_recall"] - reduced_validation["mitm_recall"]
    )
    accepted = (
        14 <= len(selected_features) <= 15
        and accuracy_drop <= 0.001
        and macro_f1_drop <= 0.002
        and mitm_recall_drop <= 0.01
    )
    decision = pd.DataFrame(
        [
            {
                "accepted": accepted,
                "selected_feature_count": len(selected_features),
                "accuracy_drop": accuracy_drop,
                "macro_f1_drop": macro_f1_drop,
                "mitm_recall_drop": mitm_recall_drop,
            }
        ]
    )
    decision.to_csv(REDUCTION_3_DIR / "acceptance_decision.csv", index=False)
    print("\n========== Reduction-3 Acceptance ==========")
    print(decision.to_string(index=False))

    if not accepted:
        print("[WARN] Reduction-3 rejected; test data was not evaluated.")
        return {
            "accepted": False,
            "selected_features": selected_features,
            "validation_comparison": validation_comparison,
        }

    metrics, report, cm, y_pred, y_prob, classwise_time_df = evaluate_model(
        reduced_model,
        X_test[selected_features],
        y_test,
    )
    save_confusion_matrix_plot(cm, CLASS_NAMES, REDUCTION_3_DIR)
    classwise_time_df.to_csv(
        REDUCTION_3_DIR / "classwise_inference_time.csv",
        index=False,
    )
    save_reports(
        metrics,
        report,
        cm,
        y_test,
        y_pred,
        y_prob,
        REDUCTION_3_DIR,
    )
    save_feature_importance(
        reduced_model,
        selected_features,
        REDUCTION_3_DIR,
    )
    save_model(
        reduced_model,
        MODEL_DIR,
        filename="lightgbm_toniot_classification_reduction_3.pkl",
    )
    return {
        "accepted": True,
        "selected_features": selected_features,
        "validation_comparison": validation_comparison,
        "test_metrics": metrics,
    }


def run_lightgbm_pipeline(save_results=True, feature_selection=False, fs_config=None, run_xai=True):
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_processed_data(
        PROCESSED_DIR,
        feature_selection=feature_selection,
        fs_config=fs_config,
    )

    # Tune the number of boosting rounds on validation, then evaluate test once.
    model = create_lightgbm_model(
        num_classes=len(CLASS_NAMES),
        random_state=RANDOM_STATE,
    )
    model = train_model(model, X_train, y_train, X_valid, y_valid)

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
            xai_result = run_xai_feature_selection(
                X_train,
                y_train,
                X_valid,
                y_valid,
            )
            print("\n[INFO] Features for the next reduction phase:")
            print(xai_result.final_features)
    else:
        print("Terminate experiment without saving results")


if __name__ == "__main__":
    run_lightgbm_pipeline(save_results=True, run_xai=True)
