"""Verify archived CSV results and build README assets; never train a model.

Run from a complete xai checkout after installing requirements-toniot.txt:
    python docs/tools/build_readme_assets.py
Only docs/research_summary.json and docs/assets/ are written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

ARCHIVE_COMMIT = "c8cc8596f22900d994eca525878c8c674e2e98e4"
CLASSES = ["backdoor", "ddos", "dos", "injection", "mitm", "normal", "password", "ransomware", "scanning", "xss"]
REPORT = Path("reports/lightgbm_260805")
STAGES = [("baseline", 22, "", "lightgbm_toniot_classification.pkl"),
          ("reduction_1", 18, "reduction_1", "lightgbm_toniot_classification_reduction_1.pkl"),
          ("reduction_2", 16, "reduction_2", "lightgbm_toniot_classification_reduction_2.pkl")]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_from_cm(cm: list[list[int]]) -> tuple[dict[str, float], list[dict]]:
    total = sum(map(sum, cm))
    per_class = []
    for i, name in enumerate(CLASSES):
        support = sum(cm[i])
        predicted = sum(row[i] for row in cm)
        tp = cm[i][i]
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(dict(label=name, precision=precision, recall=recall, f1=f1, support=support))
    return {
        "accuracy": sum(cm[i][i] for i in range(len(CLASSES))) / total,
        "precision_macro": sum(item["precision"] for item in per_class) / len(CLASSES),
        "recall_macro": sum(item["recall"] for item in per_class) / len(CLASSES),
        "f1_macro": sum(item["f1"] for item in per_class) / len(CLASSES),
    }, per_class


def collect(source: Path) -> dict:
    raw = source / "data_network/raw/train_test_network.csv"
    counts, dash_counts = Counter(), Counter()
    with raw.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        for row in reader:
            counts[row["type"]] += 1
            dash_counts.update(key for key, value in row.items() if value == "-")
    assert sorted(counts) == CLASSES
    count = sum(counts.values())
    splits = {}
    for split in ("train", "valid", "test"):
        path = source / f"data_network/processed_type/y_{split}.csv"
        items = rows(path)
        labels = Counter(int(next(iter(item.values()))) for item in items)
        splits[split] = {"rows": len(items), "counts": {name: labels[i] for i, name in enumerate(CLASSES)}}
    assert sum(part["rows"] for part in splits.values()) == count
    for name in CLASSES:
        assert sum(part["counts"][name] for part in splits.values()) == counts[name]
    with (source / "data_network/processed_type/X_train.csv").open(encoding="utf-8-sig", newline="") as handle:
        baseline_features = next(csv.reader(handle))
    selected = [row["feature"] for row in rows(source / REPORT / "xai/reduction_1/selection/final_selected_features.csv")]
    assert len(baseline_features) == 22 and len(selected) == len(set(selected)) == 18
    models = []
    for name, features, folder, filename in STAGES:
        report = source / REPORT / folder
        saved = {key: float(value) for key, value in rows(report / "metrics.csv")[0].items()}
        cm = [[int(row[str(i)]) for i in range(10)] for row in rows(report / "confusion_matrix.csv")]
        recomputed, per_class = metric_from_cm(cm)
        for key, value in recomputed.items():
            assert math.isclose(value, saved[key], abs_tol=1e-12), (name, key)
        predictions = rows(report / "predictions.csv")
        actual_cm = [[0] * 10 for _ in range(10)]
        for prediction in predictions:
            actual_cm[int(prediction["y_true"])][int(prediction["y_pred"])] += 1
        assert actual_cm == cm and len(predictions) == splits["test"]["rows"]
        assert math.isclose(saved["avg_inference_time_ms"], saved["total_inference_time_sec"] / len(predictions) * 1000, abs_tol=1e-12)
        model_file = source / "models" / filename
        models.append(dict(name=name, features=features, test_rows=len(predictions), metrics=saved,
                           per_class=per_class, confusion_matrix=cm, model_file=f"models/{filename}",
                           model_bytes=model_file.stat().st_size, sha256=digest(model_file),
                           source_report=(REPORT / folder).as_posix()))
    gates = []
    for index in (1, 2, 3):
        directory = source / REPORT / f"reduction_{index}"
        decision = rows(directory / "acceptance_decision.csv")[0]
        validation = rows(directory / "validation_comparison.csv")
        gates.append({"stage": f"reduction_{index}", "features": int(decision["selected_feature_count"]),
                      "accepted": decision["accepted"].lower() == "true",
                      "drops": {key: float(decision[key]) for key in ("accuracy_drop", "macro_f1_drop", "mitm_recall_drop")},
                      "validation": [{"model": item["model"], **{k: float(v) for k, v in item.items() if k != "model"}} for item in validation]})
    assert models[1]["sha256"] == "869871363142db6794b3a7d959553d152e5569739383601f70f8256524e684ea"
    return {
        "source_commit": ARCHIVE_COMMIT,
        "scope": "Archived CSV/model-file audit; no retraining or new timing run",
        "dataset": {"source_csv": "data_network/raw/train_test_network.csv", "sha256": digest(raw),
                    "rows": count, "columns": len(columns), "class_counts": dict(sorted(counts.items())),
                    "splits": splits, "baseline_features": baseline_features, "selected_features": selected,
                    "dash_heavy_columns": [key for key in columns if dash_counts[key] / count >= .95]},
        "models": models, "validation_gates": gates,
        "timing_scope": "Wall-clock batch predict plus predict_proba; per-row amortized cost, not online request latency",
    }


def svg_text(x, y, value, size=20, color="#e2e8f0", weight=400, anchor="start"):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{escape(str(value))}</text>'


def write_brand_assets(assets: Path) -> None:
    base = '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="360" viewBox="0 0 1000 360" role="img" aria-labelledby="title desc"><title id="title">TON IoT IDS</title><desc id="desc">Explainable LightGBM intrusion detection with 18 features and ten classes.</desc><defs><linearGradient id="bg" x2="1" y2="1"><stop stop-color="#0b1223"/><stop offset="1" stop-color="#172f50"/></linearGradient></defs><rect width="1000" height="360" rx="24" fill="url(#bg)"/><g font-family="Arial, Helvetica, sans-serif">'
    content = svg_text(44, 51, "NETWORK SECURITY  /  EXPLAINABLE MACHINE LEARNING", 17, "#5eead4", 600)
    content += svg_text(44, 112, "TON IoT IDS", 47, "#ffffff", 700)
    content += svg_text(44, 150, "LightGBM + class-aware XAI feature selection", 24, "#cbd5e1")
    for index, (value, label) in enumerate((("211,043", "network records"), ("10", "traffic classes"), ("18", "selected features"), ("97.8172%", "test accuracy"))):
        x = 44 + index * 238
        content += f'<rect x="{x}" y="192" width="220" height="119" rx="12" fill="#ffffff" fill-opacity=".06" stroke="#ffffff" stroke-opacity=".15"/>'
        content += svg_text(x + 18, 240, value, 30, "#ffffff", 700)
        content += svg_text(x + 18, 277, label, 17, "#bfdbfe")
    (assets / "hero.svg").write_text(base + content + '</g></svg>\n', encoding="utf-8")
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1110" viewBox="0 0 1000 1110" role="img" aria-labelledby="title desc"><title id="title">Experiment pipeline</title><desc id="desc">Network records are split, transformed, used for baseline training and class-aware XAI selection, then validated before test evaluation and deployment.</desc><defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0 0 L9 4.5 L0 9" fill="#64748b"/></marker></defs><rect width="1000" height="1110" rx="22" fill="#f8fafc"/><g font-family="Arial, Helvetica, sans-serif">']
    out += [svg_text(48, 53, "From network records to a deployable model", 30, "#0f172a", 700), svg_text(48, 87, "Feature selection uses training data; early stopping and acceptance use validation.", 19, "#475569")]
    cards = [
        ("01", "TON_IoT network CSV", "211,043 records / 44 columns / ten traffic classes", "#2563eb"),
        ("02", "Preprocessing and stratified split", "70% train / 15% validation / 15% test; 22 input features", "#2563eb"),
        ("03", "Baseline LightGBM + training-set XAI", "Macro-SAGE core + class-wise SHAP / approximate CPFI protection", "#7c3aed"),
        ("04", "Retrain reduced candidates", "22 -> 18 -> 16 -> 14 features; preserve attack-specific information", "#7c3aed"),
        ("05", "Validation gate, then test evaluation", "Accuracy / macro F1 / MITM recall gate; 14-feature candidate rejected", "#0f766e"),
        ("06", "18-feature model used by the companion system", "Stored model + encoders + scaler + ordered feature list", "#0f766e"),
    ]
    for i, (number, title, detail, color) in enumerate(cards):
        y = 123 + i * 153
        out += [f'<rect x="48" y="{y}" width="904" height="110" rx="15" fill="#ffffff" stroke="#cbd5e1"/><rect x="48" y="{y}" width="8" height="110" rx="4" fill="{color}"/>',
                svg_text(78, y + 48, number, 28, color, 700), svg_text(143, y + 43, title, 24, "#0f172a", 700), svg_text(143, y + 77, detail, 18, "#475569")]
        if i < 5:
            out.append(f'<path d="M500 {y+115} V{y+144}" fill="none" stroke="#64748b" stroke-width="2.2" marker-end="url(#arrow)"/>')
    out += [svg_text(48, 1070, "Network-flow classification is separate from live packet capture and device sensor telemetry.", 19, "#475569"), '</g></svg>\n']
    (assets / "pipeline.svg").write_text("".join(out), encoding="utf-8")


def write_plots(summary: dict, assets: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.labelcolor": "#334155", "text.color": "#0f172a", "axes.spines.top": False, "axes.spines.right": False})
    models = summary["models"]
    labels = ["Baseline\n22 features", "Reduction 1\n18 features", "Reduction 2\n16 features"]
    colors = ["#2563eb", "#0f766e", "#94a3b8"]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 9.4), layout="constrained")
    panels = [([m["metrics"]["f1_macro"] * 100 for m in models], "Macro F1-score (%)", (94, 96.6), ".4f"),
              ([m["model_bytes"] / 1e6 for m in models], "Stored model size (decimal MB)", (0, 9.2), ".3f"),
              ([m["metrics"]["total_inference_time_sec"] for m in models], "Batch evaluation time (seconds)", (0, 2.8), ".4f"),
              ([m["features"] for m in models], "Input feature count", (0, 26), ".0f")]
    for ax, (values, title, limits, fmt) in zip(axes.flat, panels):
        bars = ax.bar(labels, values, color=colors, width=.59, zorder=3)
        ax.set_title(title, pad=17)
        ax.set_ylim(*limits)
        ax.grid(axis="y", color="#e2e8f0", zorder=0)
        ax.bar_label(bars, labels=[format(value, fmt) for value in values], padding=5, fontsize=12)
        ax.tick_params(axis="x", length=0, pad=9)
    fig.suptitle("Archived LightGBM comparison | 31,657 test records", fontsize=18, fontweight="bold")
    fig.supxlabel("Timing includes predict + predict_proba on the full batch; not single-request latency.", fontsize=11)
    fig.savefig(assets / "model_comparison.png", dpi=175, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.4, 6.4), layout="constrained")
    values = [summary["dataset"]["class_counts"][name] for name in CLASSES]
    bars = ax.barh(CLASSES, values, color=["#d97706" if n == "mitm" else "#2563eb" for n in CLASSES], height=.65)
    ax.bar_label(bars, labels=[f"{v:,}" for v in values], padding=7)
    ax.set_xlim(0, max(values) * 1.17)
    ax.invert_yaxis()
    ax.set_xlabel("Records in this repository's network CSV")
    ax.set_title("TON_IoT network subset | 211,043 records", pad=18)
    ax.grid(axis="x", color="#e2e8f0")
    ax.set_axisbelow(True)
    fig.savefig(assets / "class_distribution.png", dpi=175, facecolor="white")
    plt.close(fig)

    cm = np.array(models[1]["confusion_matrix"])
    rates = cm / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(11.8, 10.1), layout="constrained")
    im = ax.imshow(rates, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASSES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("18-feature LightGBM | row-normalized confusion matrix", pad=18)
    for i in range(10):
        for j in range(10):
            text = f"{rates[i,j]:.1f}%" if cm[i,j] else "0"
            ax.text(j, i, text, ha="center", va="center", fontsize=10.5, color="white" if rates[i,j] > 55 else "#0f172a")
    fig.colorbar(im, ax=ax, shrink=.79, label="Share of the true class (%)")
    fig.savefig(assets / "confusion_matrix_normalized.png", dpi=175, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    summary = collect(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    assets = args.output / "assets"
    assets.mkdir(exist_ok=True)
    (args.output / "research_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_brand_assets(assets)
    if not args.skip_plots:
        write_plots(summary, assets)
    print("Verified all three saved prediction/CM/metric sets; generated aggregate documentation assets.")


if __name__ == "__main__":
    main()
