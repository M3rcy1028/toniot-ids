from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit


@dataclass(slots=True)
class XAIConfig:
    """Configuration for SAGE-led full-train feature selection."""

    random_state: int = 42

    # Full-train computations are batched to control memory usage.
    shap_batch_size: int = 4096
    interaction_batch_size: int = 256
    sage_repeats: int = 1
    cpf_max_samples: int | None = None
    correlation_max_samples: int | None = None

    # Class-wise conditional permutation feature importance (CPFI).
    cpf_metric: str = "f1"  # one of: f1, precision, recall
    cpf_repeats: int = 5
    cpf_n_conditioners: int = 2
    cpf_n_bins: int = 5

    # Consensus selection.
    top_k_sage: int = 14
    top_k_shap: int = 12
    top_k_cpf: int = 12
    min_features_per_class: int = 3
    min_final_features: int = 18
    max_final_features: int = 20

    # SHAP interaction protection.
    enable_interaction: bool = True  # safely auto-skipped if unsupported
    top_interaction_pairs_per_class: int = 5
    interaction_add_orphan_pairs: bool = False

    # LIME is best treated as a local error-analysis aid.
    enable_lime: bool = False
    lime_samples_per_class: int = 3
    lime_num_features: int = 10
    lime_num_samples: int = 2000
    lime_training_max_samples: int = 5000
    include_lime_in_selection: bool = False
    top_lime_features_per_class: int = 3

    # Reporting.
    plot_top_n: int = 20


@dataclass(slots=True)
class XAIResult:
    global_sage: pd.DataFrame
    classwise_sage: pd.DataFrame
    global_shap: pd.DataFrame
    classwise_shap: pd.DataFrame
    global_cpf: pd.DataFrame
    classwise_cpf: pd.DataFrame
    interactions: pd.DataFrame
    lime_local: pd.DataFrame
    lime_aggregate: pd.DataFrame
    selection_scores: pd.DataFrame
    classwise_selected: pd.DataFrame
    final_features: list[str]


def _ensure_numeric_frame(X: pd.DataFrame, name: str) -> None:
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise TypeError(
            f"{name} contains non-numeric columns: {non_numeric}. "
            "Encode categorical columns before running XAI analysis."
        )
    if X.columns.duplicated().any():
        duplicates = X.columns[X.columns.duplicated()].tolist()
        raise ValueError(f"{name} contains duplicate feature names: {duplicates}")


def _stratified_positions(
    y: pd.Series,
    max_samples: int | None,
    random_state: int,
) -> np.ndarray:
    n_rows = len(y)
    if max_samples is None or max_samples >= n_rows:
        return np.arange(n_rows)
    if max_samples <= 0:
        raise ValueError("max_samples must be positive or None")

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=max_samples,
        random_state=random_state,
    )
    dummy = np.zeros((n_rows, 1), dtype=np.uint8)
    try:
        positions, _ = next(splitter.split(dummy, y.to_numpy()))
        return np.sort(positions)
    except ValueError:
        rng = np.random.default_rng(random_state)
        return np.sort(rng.choice(n_rows, size=max_samples, replace=False))


def _normalize_shap_values(
    values: Any,
    n_samples: int,
    n_features: int,
    n_outputs: int,
) -> np.ndarray:
    """Return SHAP values as (samples, features, outputs)."""
    if isinstance(values, list):
        arrays = [np.asarray(value) for value in values]
        return np.stack(arrays, axis=-1)

    array = np.asarray(values)
    if array.ndim == 2:
        return array[:, :, np.newaxis]

    if array.ndim != 3:
        raise ValueError(f"Unexpected SHAP value shape: {array.shape}")

    if array.shape[0] == n_samples and array.shape[1] == n_features:
        return array
    if (
        array.shape[0] == n_outputs
        and array.shape[1] == n_samples
        and array.shape[2] == n_features
    ):
        return np.moveaxis(array, 0, -1)
    if (
        array.shape[0] == n_samples
        and array.shape[1] == n_outputs
        and array.shape[2] == n_features
    ):
        return np.moveaxis(array, 1, -1)

    raise ValueError(f"Cannot normalize SHAP value shape: {array.shape}")


def _normalize_interaction_values(
    values: Any,
    n_samples: int,
    n_features: int,
    n_outputs: int,
) -> np.ndarray:
    """Normalize SHAP interaction values to a four-dimensional array.

    The returned layout is::

        (n_samples, n_features, n_features, interaction_outputs)

    ``interaction_outputs`` is not forcibly matched to ``n_outputs`` here.
    Some SHAP/LightGBM combinations return only one interaction tensor for a
    multiclass model. Such an output must not be duplicated and presented as
    class-specific interactions. The caller decides whether class-wise
    interaction analysis is available.
    """
    if n_samples <= 0 or n_features <= 0 or n_outputs <= 0:
        raise ValueError(
            "n_samples, n_features, and n_outputs must all be positive"
        )

    expected_matrix_shape = (n_samples, n_features, n_features)

    if isinstance(values, (list, tuple)):
        if not values:
            raise ValueError("SHAP returned an empty interaction list")

        normalized_arrays: list[np.ndarray] = []

        for output_position, value in enumerate(values):
            array = np.asarray(value)

            if array.shape != expected_matrix_shape:
                raise ValueError(
                    "Unexpected SHAP interaction shape in output "
                    f"{output_position}: {array.shape}; expected "
                    f"{expected_matrix_shape}"
                )

            normalized_arrays.append(array)

        return np.stack(normalized_arrays, axis=-1)

    array = np.asarray(values)

    if array.ndim == 3:
        if array.shape != expected_matrix_shape:
            raise ValueError(
                "Unexpected three-dimensional SHAP interaction shape: "
                f"{array.shape}; expected {expected_matrix_shape}"
            )

        return array[..., np.newaxis]

    if array.ndim != 4:
        raise ValueError(
            f"Unexpected SHAP interaction shape: {array.shape}"
        )

    normalized: np.ndarray | None = None

    if (
        array.shape[0] == n_samples
        and array.shape[1] == n_features
        and array.shape[2] == n_features
    ):
        normalized = array

    elif (
        array.shape[1] == n_samples
        and array.shape[2] == n_features
        and array.shape[3] == n_features
    ):
        normalized = np.moveaxis(array, 0, -1)

    elif (
        array.shape[0] == n_samples
        and array.shape[2] == n_features
        and array.shape[3] == n_features
    ):
        normalized = np.moveaxis(array, 1, -1)

    elif (
        array.shape[0] == n_samples
        and array.shape[1] == n_features
        and array.shape[3] == n_features
    ):
        normalized = np.moveaxis(array, 2, -1)

    if normalized is None:
        raise ValueError(
            "Cannot normalize SHAP interaction shape: "
            f"{array.shape}. Expected a permutation of "
            "(samples, features, features, outputs)."
        )

    if normalized.shape[:3] != expected_matrix_shape:
        raise ValueError(
            "Normalized SHAP interaction tensor has invalid leading "
            f"dimensions: {normalized.shape}"
        )

    return normalized


def _empty_interaction_frame() -> pd.DataFrame:
    """Return an empty interaction report with a stable CSV schema."""
    return pd.DataFrame(
        columns=[
            "class_id",
            "class_name",
            "feature_1",
            "feature_2",
            "mean_abs_interaction",
            "rank_interaction",
            "class_samples",
        ]
    )


def _make_tree_explainer(model: Any):
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "The 'shap' package is required. Install it with: pip install shap"
        ) from exc

    tree_model = getattr(model, "booster_", model)
    return shap.TreeExplainer(
        tree_model,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )


def compute_classwise_shap_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    class_ids: Sequence[Any],
    class_names: Sequence[str],
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    """Compute SHAP importance for every training row in memory-safe batches."""
    if batch_size <= 0:
        raise ValueError("shap_batch_size must be positive")

    explainer = _make_tree_explainer(model)
    n_features = X.shape[1]
    n_classes = len(class_ids)
    class_to_position = {class_id: i for i, class_id in enumerate(class_ids)}
    global_sum = np.zeros(n_features, dtype=float)
    class_sums = np.zeros((n_classes, n_features), dtype=float)
    class_counts = np.zeros(n_classes, dtype=int)

    print(f"[INFO] Computing full-train SHAP for {len(X):,} rows...")
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        X_batch = X.iloc[start:stop]
        y_batch = y.iloc[start:stop].to_numpy()
        try:
            raw_values = explainer.shap_values(X_batch, check_additivity=False)
        except TypeError:
            raw_values = explainer.shap_values(X_batch)

        values = _normalize_shap_values(
            raw_values,
            n_samples=len(X_batch),
            n_features=n_features,
            n_outputs=n_classes,
        )
        if values.shape[-1] != n_classes:
            raise ValueError(
                "The number of SHAP output dimensions does not match "
                f"model.classes_: outputs={values.shape[-1]}, classes={n_classes}"
            )

        absolute = np.abs(values)
        global_sum += absolute.sum(axis=(0, 2))
        for class_id, class_position in class_to_position.items():
            mask = y_batch == class_id
            if np.any(mask):
                class_sums[class_position] += absolute[
                    mask, :, class_position
                ].sum(axis=0)
                class_counts[class_position] += int(mask.sum())

        if stop == len(X) or stop % (batch_size * 10) == 0:
            print(f"[INFO] SHAP progress: {stop:,}/{len(X):,}")

    global_values = global_sum / (len(X) * n_classes)
    global_df = pd.DataFrame(
        {
            "feature": X.columns,
            "mean_abs_shap": global_values,
        }
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    global_df["rank"] = np.arange(1, len(global_df) + 1)

    rows: list[dict[str, Any]] = []
    for output_position, (class_id, class_name) in enumerate(
        zip(class_ids, class_names, strict=True)
    ):
        if class_counts[output_position] == 0:
            warnings.warn(f"No SHAP training row exists for class {class_id!r}")
            continue
        class_importance = (
            class_sums[output_position] / class_counts[output_position]
        )
        order = np.argsort(-class_importance)
        for rank, feature_position in enumerate(order, start=1):
            rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "feature": X.columns[feature_position],
                    "mean_abs_shap": float(class_importance[feature_position]),
                    "rank_shap": rank,
                    "class_samples": int(class_counts[output_position]),
                }
            )

    return global_df, pd.DataFrame(rows), explainer


def compute_classwise_sage_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    class_ids: Sequence[Any],
    class_names: Sequence[str],
    repeats: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate class-balanced SAGE with every training row.

    This follows the permutation SAGE estimator, but replaces its default
    bootstrap row sampling with full passes over the training set. Missing
    features are marginalized with a random donor permutation of the same
    training set. Each class uses a balanced one-vs-rest binary cross-entropy
    loss, so minority and majority classes receive equal weight.
    """
    try:
        import sage  # noqa: F401 - dependency and method provenance check
    except ImportError as exc:
        raise ImportError(
            "SAGE selection requires 'sage-importance'. Install it with: "
            "pip install sage-importance"
        ) from exc

    if repeats <= 0:
        raise ValueError("sage_repeats must be positive")

    X_values = X.to_numpy(copy=True)
    y_values = y.to_numpy()
    n_rows, n_features = X_values.shape
    n_classes = len(class_ids)
    class_to_position = {class_id: i for i, class_id in enumerate(class_ids)}
    try:
        y_positions = np.fromiter(
            (class_to_position[value] for value in y_values),
            dtype=int,
            count=n_rows,
        )
    except KeyError as exc:
        raise ValueError(f"Unknown class id in y_reference: {exc.args[0]!r}") from exc

    targets = np.equal.outer(y_positions, np.arange(n_classes)).astype(float)
    positive_counts = targets.sum(axis=0)
    negative_counts = n_rows - positive_counts
    if np.any(positive_counts == 0) or np.any(negative_counts == 0):
        raise ValueError("Each SAGE class requires positive and negative rows")
    weights = np.where(
        targets == 1,
        0.5 / positive_counts,
        0.5 / negative_counts,
    )

    rng = np.random.default_rng(random_state)
    repeat_values = np.zeros((repeats, n_classes, n_features), dtype=float)
    row_positions = np.arange(n_rows)

    def predict_probabilities(values: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(values, columns=X.columns, copy=False)
        return np.asarray(model.predict_proba(frame))

    def balanced_binary_losses(probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
        return -(
            targets * np.log(clipped)
            + (1 - targets) * np.log1p(-clipped)
        )

    print(
        f"[INFO] Computing full-train class-balanced SAGE for "
        f"{n_rows:,} rows and {n_features} features..."
    )
    for repeat in range(repeats):
        donor_positions = rng.permutation(n_rows)
        revealed = X_values[donor_positions].copy()
        feature_orders = np.argsort(
            rng.random((n_rows, n_features)),
            axis=1,
        )
        previous_loss = balanced_binary_losses(predict_probabilities(revealed))

        for step in range(n_features):
            revealed_features = feature_orders[:, step]
            revealed[row_positions, revealed_features] = X_values[
                row_positions, revealed_features
            ]
            current_loss = balanced_binary_losses(predict_probabilities(revealed))
            contributions = (previous_loss - current_loss) * weights

            for class_position in range(n_classes):
                np.add.at(
                    repeat_values[repeat, class_position],
                    revealed_features,
                    contributions[:, class_position],
                )
            previous_loss = current_loss

            if step == n_features - 1 or (step + 1) % 5 == 0:
                print(
                    f"[INFO] SAGE repeat {repeat + 1}/{repeats}: "
                    f"{step + 1}/{n_features} features revealed"
                )

    class_values = repeat_values.mean(axis=0)
    class_std = repeat_values.std(axis=0, ddof=0)
    class_rows: list[dict[str, Any]] = []
    for class_position, (class_id, class_name) in enumerate(
        zip(class_ids, class_names, strict=True)
    ):
        order = np.argsort(-class_values[class_position])
        for rank, feature_position in enumerate(order, start=1):
            class_rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "feature": X.columns[feature_position],
                    "sage_importance": float(
                        class_values[class_position, feature_position]
                    ),
                    "sage_repeat_std": float(
                        class_std[class_position, feature_position]
                    ),
                    "rank_sage": rank,
                    "positive_samples": int(positive_counts[class_position]),
                    "negative_samples": int(negative_counts[class_position]),
                    "repeats": repeats,
                }
            )

    macro_values = class_values.mean(axis=0)
    macro_std = repeat_values.mean(axis=1).std(axis=0, ddof=0)
    global_df = pd.DataFrame(
        {
            "feature": X.columns,
            "macro_sage_importance": macro_values,
            "sage_repeat_std": macro_std,
        }
    ).sort_values("macro_sage_importance", ascending=False, ignore_index=True)
    global_df["rank_sage"] = np.arange(1, len(global_df) + 1)
    return global_df, pd.DataFrame(class_rows)


def _metric_per_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_ids: Sequence[Any],
    metric: str,
) -> np.ndarray:
    common = {
        "labels": list(class_ids),
        "average": None,
        "zero_division": 0,
    }
    if metric == "f1":
        return f1_score(y_true, y_pred, **common)
    if metric == "precision":
        return precision_score(y_true, y_pred, **common)
    if metric == "recall":
        return recall_score(y_true, y_pred, **common)
    raise ValueError("cpf_metric must be one of: 'f1', 'precision', 'recall'")


def _conditioning_features(
    X_reference: pd.DataFrame,
    n_conditioners: int,
    max_samples: int | None,
    random_state: int,
) -> dict[str, list[str]]:
    if n_conditioners <= 0:
        return {feature: [] for feature in X_reference.columns}

    if max_samples is not None and len(X_reference) > max_samples:
        X_corr = X_reference.sample(
            n=max_samples,
            random_state=random_state,
        )
    else:
        X_corr = X_reference

    correlation = X_corr.corr(method="spearman").abs().fillna(0.0)
    result: dict[str, list[str]] = {}
    for feature in X_reference.columns:
        ranked = correlation[feature].drop(index=feature).sort_values(ascending=False)
        ranked = ranked[ranked > 0]
        result[feature] = ranked.head(n_conditioners).index.tolist()
    return result


def _group_codes(
    X: pd.DataFrame,
    conditioners: Sequence[str],
    n_bins: int,
) -> np.ndarray | None:
    if not conditioners:
        return None

    code_columns: dict[str, pd.Series] = {}
    for conditioner in conditioners:
        series = X[conditioner]
        unique_count = int(series.nunique(dropna=True))
        if unique_count < 2:
            continue
        q = min(n_bins, unique_count)
        try:
            code_columns[conditioner] = pd.qcut(
                series,
                q=q,
                labels=False,
                duplicates="drop",
            ).fillna(-1)
        except ValueError:
            continue

    if not code_columns:
        return None

    code_frame = pd.DataFrame(code_columns, index=X.index).astype(int)
    codes = pd.factorize(
        pd.MultiIndex.from_frame(code_frame),
        sort=False,
    )[0]

    counts = pd.Series(codes).value_counts()
    singleton_fraction = float((counts == 1).sum() / max(len(counts), 1))
    if singleton_fraction > 0.5 and len(code_columns) > 1:
        first_conditioner = next(iter(code_columns))
        return _group_codes(X, [first_conditioner], n_bins)
    return codes


def _shuffle_within_groups(
    values: np.ndarray,
    group_codes: np.ndarray | None,
    rng: np.random.Generator,
) -> np.ndarray:
    shuffled = np.array(values, copy=True)
    if group_codes is None:
        return rng.permutation(shuffled)

    for group_id in np.unique(group_codes):
        positions = np.flatnonzero(group_codes == group_id)
        if positions.size > 1:
            shuffled[positions] = rng.permutation(shuffled[positions])
    return shuffled


def compute_classwise_conditional_permutation_importance(
    model: Any,
    X_reference: pd.DataFrame,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    class_ids: Sequence[Any],
    class_names: Sequence[str],
    config: XAIConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Approximate class-wise conditional permutation importance.

    A feature is shuffled only within bins formed by its strongest correlated
    conditioning features. This is a practical CPFI approximation rather than
    an exact sample from the full conditional distribution p(X_j | X_-j).
    """
    positions = _stratified_positions(
        y_eval,
        config.cpf_max_samples,
        config.random_state,
    )
    X_sample = X_eval.iloc[positions].copy().reset_index(drop=True)
    y_sample = y_eval.iloc[positions].reset_index(drop=True)

    conditioners_by_feature = _conditioning_features(
        X_reference,
        n_conditioners=config.cpf_n_conditioners,
        max_samples=config.correlation_max_samples,
        random_state=config.random_state,
    )

    baseline_pred = model.predict(X_sample)
    baseline_scores = _metric_per_class(
        y_sample.to_numpy(),
        np.asarray(baseline_pred),
        class_ids,
        config.cpf_metric,
    )

    rng = np.random.default_rng(config.random_state)
    rows: list[dict[str, Any]] = []

    for feature_position, feature in enumerate(X_sample.columns, start=1):
        conditioners = conditioners_by_feature[feature]
        groups = _group_codes(X_sample, conditioners, config.cpf_n_bins)
        repeat_scores: list[np.ndarray] = []

        for _ in range(config.cpf_repeats):
            X_permuted = X_sample.copy()
            X_permuted[feature] = _shuffle_within_groups(
                X_sample[feature].to_numpy(),
                groups,
                rng,
            )
            permuted_pred = model.predict(X_permuted)
            repeat_scores.append(
                _metric_per_class(
                    y_sample.to_numpy(),
                    np.asarray(permuted_pred),
                    class_ids,
                    config.cpf_metric,
                )
            )

        repeated = np.stack(repeat_scores, axis=0)
        importance = baseline_scores[np.newaxis, :] - repeated

        for class_position, (class_id, class_name) in enumerate(
            zip(class_ids, class_names, strict=True)
        ):
            rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "feature": feature,
                    "metric": config.cpf_metric,
                    "baseline_score": float(baseline_scores[class_position]),
                    "permuted_score_mean": float(repeated[:, class_position].mean()),
                    "cpf_importance_mean": float(importance[:, class_position].mean()),
                    "cpf_importance_std": float(importance[:, class_position].std(ddof=0)),
                    "conditioning_features": ";".join(conditioners),
                    "repeats": config.cpf_repeats,
                }
            )

        print(
            f"[INFO] CPF progress: {feature_position}/{X_sample.shape[1]} "
            f"({feature})"
        )

    classwise_df = pd.DataFrame(rows)
    classwise_df["rank_cpf"] = (
        classwise_df.groupby("class_id")["cpf_importance_mean"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    classwise_df = classwise_df.sort_values(
        ["class_id", "rank_cpf"],
        ignore_index=True,
    )

    global_df = (
        classwise_df.groupby("feature", as_index=False)
        .agg(
            mean_classwise_cpf=("cpf_importance_mean", "mean"),
            max_classwise_cpf=("cpf_importance_mean", "max"),
            positive_class_count=(
                "cpf_importance_mean",
                lambda values: int((values > 0).sum()),
            ),
        )
        .sort_values("mean_classwise_cpf", ascending=False, ignore_index=True)
    )
    global_df["rank"] = np.arange(1, len(global_df) + 1)
    return global_df, classwise_df


def compute_classwise_shap_interactions(
    explainer: Any,
    X: pd.DataFrame,
    y: pd.Series,
    class_ids: Sequence[Any],
    class_names: Sequence[str],
    batch_size: int,
) -> pd.DataFrame:
    """Compute full-train class-wise SHAP interactions in batches.

    A single interaction tensor returned for a multiclass model is not copied
    across classes because that would create artificial class-wise results.
    In that case, interaction-based feature protection is skipped while the
    class-wise SHAP and CPFI stages continue normally.
    """
    if batch_size <= 0:
        raise ValueError("interaction_batch_size must be positive")

    n_features = X.shape[1]
    n_classes = len(class_ids)
    class_to_position = {class_id: i for i, class_id in enumerate(class_ids)}
    class_sums = np.zeros(
        (n_classes, n_features, n_features),
        dtype=float,
    )
    class_counts = np.zeros(n_classes, dtype=int)

    print(f"[INFO] Computing full-train SHAP interactions for {len(X):,} rows...")
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        X_batch = X.iloc[start:stop]
        y_batch = y.iloc[start:stop].to_numpy()
        try:
            raw_values = explainer.shap_interaction_values(X_batch)
            values = _normalize_interaction_values(
                raw_values,
                n_samples=len(X_batch),
                n_features=n_features,
                n_outputs=n_classes,
            )
        except Exception as error:
            print(
                "[INFO] SHAP interaction analysis skipped because interaction "
                f"values could not be computed safely: {type(error).__name__}: "
                f"{error}"
            )
            return _empty_interaction_frame()

        if values.shape[-1] != n_classes:
            print(
                "[INFO] SHAP interaction analysis skipped: the explainer "
                f"returned {values.shape[-1]} interaction output(s) for a "
                f"{n_classes}-class model. The tensor will not be repeated "
                "because doing so would produce invalid class-wise interactions."
            )
            return _empty_interaction_frame()

        absolute = np.abs(values)
        for class_id, class_position in class_to_position.items():
            mask = y_batch == class_id
            if np.any(mask):
                class_sums[class_position] += absolute[
                    mask, :, :, class_position
                ].sum(axis=0)
                class_counts[class_position] += int(mask.sum())

        if stop == len(X) or stop % (batch_size * 20) == 0:
            print(f"[INFO] Interaction progress: {stop:,}/{len(X):,}")

    rows: list[dict[str, Any]] = []
    features = list(X.columns)
    for output_position, (class_id, class_name) in enumerate(
        zip(class_ids, class_names, strict=True)
    ):
        if class_counts[output_position] == 0:
            continue
        matrix = class_sums[output_position] / class_counts[output_position]

        pairs: list[tuple[float, int, int]] = []

        for first in range(len(features)):
            for second in range(first + 1, len(features)):
                pairs.append(
                    (float(matrix[first, second]), first, second)
                )

        pairs.sort(reverse=True, key=lambda item: item[0])

        for rank, (importance, first, second) in enumerate(
            pairs,
            start=1,
        ):
            rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "feature_1": features[first],
                    "feature_2": features[second],
                    "mean_abs_interaction": importance,
                    "rank_interaction": rank,
                    "class_samples": int(class_counts[output_position]),
                }
            )

    if not rows:
        return _empty_interaction_frame()

    return pd.DataFrame(rows)


def compute_lime_reports(
    model: Any,
    X_reference: pd.DataFrame,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    class_ids: Sequence[Any],
    class_names: Sequence[str],
    config: XAIConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as exc:
        raise ImportError(
            "LIME was enabled but the 'lime' package is missing. "
            "Install it with: pip install lime"
        ) from exc

    rng = np.random.default_rng(config.random_state)
    lime_train_size = min(config.lime_training_max_samples, len(X_reference))
    if lime_train_size < len(X_reference):
        train_positions = np.sort(
            rng.choice(len(X_reference), size=lime_train_size, replace=False)
        )
    else:
        train_positions = np.arange(len(X_reference))
    X_lime_train = X_reference.iloc[train_positions]

    explainer = LimeTabularExplainer(
        training_data=X_lime_train.to_numpy(),
        feature_names=list(X_reference.columns),
        class_names=list(class_names),
        mode="classification",
        discretize_continuous=True,
        random_state=config.random_state,
    )

    y_pred = np.asarray(model.predict(X_eval))
    y_true = y_eval.to_numpy()
    class_to_position = {class_id: i for i, class_id in enumerate(class_ids)}
    rng = np.random.default_rng(config.random_state)

    selected_positions: list[int] = []
    for class_id in class_ids:
        class_positions = np.flatnonzero(y_true == class_id)
        misclassified = class_positions[y_pred[class_positions] != class_id]
        correct = class_positions[y_pred[class_positions] == class_id]
        rng.shuffle(misclassified)
        rng.shuffle(correct)
        chosen = np.concatenate([misclassified, correct])[
            : config.lime_samples_per_class
        ]
        selected_positions.extend(chosen.tolist())

    rows: list[dict[str, Any]] = []
    for position in selected_positions:
        true_id = y_true[position]
        predicted_id = y_pred[position]
        labels_to_explain = {
            class_to_position[true_id],
            class_to_position[predicted_id],
        }
        explanation = explainer.explain_instance(
            X_eval.iloc[position].to_numpy(),
            model.predict_proba,
            labels=sorted(labels_to_explain),
            num_features=min(config.lime_num_features, X_eval.shape[1]),
            num_samples=config.lime_num_samples,
        )

        for output_position in labels_to_explain:
            explained_id = class_ids[output_position]
            explained_role = (
                "true_and_predicted"
                if true_id == predicted_id == explained_id
                else "true"
                if explained_id == true_id
                else "predicted"
            )
            for feature_position, weight in explanation.as_map()[output_position]:
                rows.append(
                    {
                        "sample_position": int(position),
                        "sample_index": X_eval.index[position],
                        "true_class_id": true_id,
                        "true_class_name": class_names[class_to_position[true_id]],
                        "predicted_class_id": predicted_id,
                        "predicted_class_name": class_names[
                            class_to_position[predicted_id]
                        ],
                        "explained_class_id": explained_id,
                        "explained_class_name": class_names[output_position],
                        "explained_role": explained_role,
                        "feature": X_eval.columns[feature_position],
                        "weight": float(weight),
                        "abs_weight": float(abs(weight)),
                    }
                )

    local_df = pd.DataFrame(rows)
    if local_df.empty:
        return local_df, pd.DataFrame()

    aggregate_df = (
        local_df.groupby(
            ["explained_class_id", "explained_class_name", "feature"],
            as_index=False,
        )
        .agg(
            mean_abs_lime=("abs_weight", "mean"),
            explanation_count=("abs_weight", "size"),
        )
    )
    aggregate_df["rank_lime"] = (
        aggregate_df.groupby("explained_class_id")["mean_abs_lime"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    aggregate_df = aggregate_df.sort_values(
        ["explained_class_id", "rank_lime"],
        ignore_index=True,
    )
    return local_df, aggregate_df


def build_consensus_selection(
    feature_order: Sequence[str],
    class_ids: Sequence[Any],
    class_names: Sequence[str],
    global_sage: pd.DataFrame,
    classwise_sage: pd.DataFrame,
    classwise_shap: pd.DataFrame,
    classwise_cpf: pd.DataFrame,
    interactions: pd.DataFrame,
    lime_aggregate: pd.DataFrame,
    config: XAIConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Select a SAGE core, then add class and interaction protection."""
    score_rows: list[pd.DataFrame] = []
    class_candidates: dict[Any, list[str]] = {}
    strict_by_class: dict[Any, set[str]] = {}
    consensus_lookup: dict[tuple[Any, str], float] = {}
    class_name_lookup = dict(zip(class_ids, class_names, strict=True))
    macro_sage = global_sage.set_index("feature")["macro_sage_importance"]
    macro_rank = global_sage.set_index("feature")["rank_sage"]

    for class_id, class_name in zip(class_ids, class_names, strict=True):
        sage_part = classwise_sage[classwise_sage["class_id"] == class_id][
            ["feature", "sage_importance", "rank_sage"]
        ]
        shap_part = classwise_shap[classwise_shap["class_id"] == class_id][
            ["feature", "mean_abs_shap", "rank_shap"]
        ]
        cpf_part = classwise_cpf[classwise_cpf["class_id"] == class_id][
            ["feature", "cpf_importance_mean", "rank_cpf"]
        ]
        merged = pd.DataFrame({"feature": list(feature_order)})
        merged = merged.merge(sage_part, on="feature", how="left")
        merged = merged.merge(shap_part, on="feature", how="left")
        merged = merged.merge(cpf_part, on="feature", how="left")
        merged["macro_sage_importance"] = merged["feature"].map(macro_sage)
        merged["macro_sage_rank"] = merged["feature"].map(macro_rank)
        merged["class_id"] = class_id
        merged["class_name"] = class_name
        merged["shap_top_k"] = merged["rank_shap"] <= config.top_k_shap
        merged["cpf_top_k"] = (
            (merged["rank_cpf"] <= config.top_k_cpf)
            & (merged["cpf_importance_mean"] > 0)
        )
        merged["strict_intersection"] = merged["shap_top_k"] & merged["cpf_top_k"]

        shap_max = float(merged["mean_abs_shap"].max() or 0.0)
        cpf_positive = merged["cpf_importance_mean"].clip(lower=0)
        cpf_max = float(cpf_positive.max() or 0.0)
        merged["shap_normalized"] = (
            merged["mean_abs_shap"] / shap_max if shap_max > 0 else 0.0
        )
        merged["cpf_normalized"] = (
            cpf_positive / cpf_max if cpf_max > 0 else 0.0
        )
        merged["consensus_score"] = (
            0.5 * merged["shap_normalized"] + 0.5 * merged["cpf_normalized"]
        )
        strict = merged.loc[merged["strict_intersection"]].sort_values(
            "consensus_score",
            ascending=False,
        )["feature"].tolist()
        candidates = list(strict)
        if len(candidates) < config.min_features_per_class:
            for feature in merged.sort_values(
                "consensus_score",
                ascending=False,
            )["feature"]:
                if feature not in candidates:
                    candidates.append(feature)
                if len(candidates) >= config.min_features_per_class:
                    break
        class_candidates[class_id] = candidates
        strict_by_class[class_id] = set(strict)
        for row in merged.itertuples(index=False):
            consensus_lookup[(class_id, row.feature)] = float(row.consensus_score)
        score_rows.append(merged)

    scores_df = pd.concat(score_rows, ignore_index=True)
    selected_sources: dict[str, set[str]] = {}
    protected_classes: dict[str, set[str]] = {}

    for feature in global_sage.head(config.top_k_sage)["feature"]:
        selected_sources.setdefault(feature, set()).add("macro_sage_core")

    def coverage(class_id: Any) -> int:
        return sum(
            feature in selected_sources
            for feature in class_candidates[class_id]
        )

    while len(selected_sources) < config.max_final_features:
        deficient = [
            class_id
            for class_id in class_ids
            if coverage(class_id) < config.min_features_per_class
        ]
        if not deficient:
            break

        ranked_candidates: list[tuple[int, float, float, str]] = []
        for feature in feature_order:
            if feature in selected_sources:
                continue
            covered = [
                class_id
                for class_id in deficient
                if feature in class_candidates[class_id]
            ]
            if not covered:
                continue
            consensus = float(
                np.mean(
                    [consensus_lookup[(class_id, feature)] for class_id in covered]
                )
            )
            ranked_candidates.append(
                (
                    len(covered),
                    consensus,
                    float(macro_sage.get(feature, -np.inf)),
                    feature,
                )
            )
        if not ranked_candidates:
            break
        _, _, _, selected_feature = max(ranked_candidates)
        selected_sources.setdefault(selected_feature, set()).add(
            "classwise_shap_cpf_protection"
        )

    for class_id in class_ids:
        class_name = class_name_lookup[class_id]
        for feature in class_candidates[class_id]:
            if feature not in selected_sources:
                continue
            protected_classes.setdefault(feature, set()).add(class_name)
            if feature in strict_by_class[class_id]:
                selected_sources[feature].add("shap_intersection_cpf")
            else:
                selected_sources[feature].add("class_consensus_fill")

    if len(selected_sources) < config.min_final_features:
        for feature in global_sage["feature"]:
            if feature not in selected_sources:
                selected_sources[feature] = {"macro_sage_floor"}
            if len(selected_sources) >= config.min_final_features:
                break

    if config.enable_interaction and not interactions.empty:
        pair_candidates = interactions[
            interactions["rank_interaction"]
            <= config.top_interaction_pairs_per_class
        ].sort_values("mean_abs_interaction", ascending=False)
        for row in pair_candidates.itertuples(index=False):
            if len(selected_sources) >= config.max_final_features:
                break
            first_selected = row.feature_1 in selected_sources
            second_selected = row.feature_2 in selected_sources
            if first_selected == second_selected:
                continue
            partner = row.feature_2 if first_selected else row.feature_1
            selected_sources.setdefault(partner, set()).add(
                "shap_interaction_partner"
            )
            protected_classes.setdefault(partner, set()).add(row.class_name)

    if (
        config.enable_lime
        and config.include_lime_in_selection
        and not lime_aggregate.empty
    ):
        for feature in lime_aggregate.sort_values(
            "mean_abs_lime",
            ascending=False,
        )["feature"]:
            if feature in selected_sources:
                selected_sources[feature].add("lime")

    uncovered = {
        class_name_lookup[class_id]: coverage(class_id)
        for class_id in class_ids
        if coverage(class_id) < config.min_features_per_class
    }
    if uncovered:
        warnings.warn(
            "The maximum feature budget prevented full class protection: "
            f"{uncovered}"
        )

    selected_rows = []
    for feature in global_sage["feature"]:
        if feature not in selected_sources:
            continue
        selected_rows.append(
            {
                "feature": feature,
                "selection_sources": ";".join(sorted(selected_sources[feature])),
                "protected_classes": ";".join(
                    sorted(protected_classes.get(feature, set()))
                ),
                "macro_sage_importance": float(macro_sage[feature]),
                "rank_sage": int(macro_rank[feature]),
            }
        )
    selected_df = pd.DataFrame(selected_rows)
    selected_set = set(selected_sources)
    final_features = [feature for feature in feature_order if feature in selected_set]
    return scores_df, selected_df, final_features


def _save_bar_plot(
    frame: pd.DataFrame,
    label_column: str,
    value_column: str,
    output_path: Path,
    title: str,
    top_n: int,
) -> None:
    if frame.empty:
        return
    plot_frame = frame.nlargest(top_n, value_column).sort_values(value_column)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(plot_frame))))
    ax.barh(plot_frame[label_column], plot_frame[value_column])
    ax.set_title(title)
    ax.set_xlabel(value_column)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_xai_reports(
    result: XAIResult,
    output_dir: Path,
    config: XAIConfig,
) -> None:
    sage_dir = output_dir / "sage"
    shap_dir = output_dir / "shap"
    cpf_dir = output_dir / "cpf"
    interaction_dir = output_dir / "interaction"
    lime_dir = output_dir / "lime"
    selection_dir = output_dir / "selection"
    for directory in [
        sage_dir,
        shap_dir,
        cpf_dir,
        interaction_dir,
        lime_dir,
        selection_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    result.global_sage.to_csv(sage_dir / "macro_sage_importance.csv", index=False)
    result.classwise_sage.to_csv(
        sage_dir / "classwise_sage_importance.csv",
        index=False,
    )
    result.global_shap.to_csv(shap_dir / "global_shap_importance.csv", index=False)
    result.classwise_shap.to_csv(
        shap_dir / "classwise_shap_importance.csv",
        index=False,
    )
    result.global_cpf.to_csv(cpf_dir / "global_cpf_importance.csv", index=False)
    result.classwise_cpf.to_csv(
        cpf_dir / "classwise_cpf_importance.csv",
        index=False,
    )
    result.interactions.to_csv(
        interaction_dir / "classwise_shap_interactions.csv",
        index=False,
    )
    result.lime_local.to_csv(lime_dir / "lime_local_explanations.csv", index=False)
    result.lime_aggregate.to_csv(
        lime_dir / "classwise_lime_importance.csv",
        index=False,
    )
    result.selection_scores.to_csv(
        selection_dir / "classwise_feature_selection_scores.csv",
        index=False,
    )
    result.classwise_selected.to_csv(
        selection_dir / "classwise_selected_features.csv",
        index=False,
    )
    pd.DataFrame({"feature": result.final_features}).to_csv(
        selection_dir / "final_selected_features.csv",
        index=False,
    )
    (selection_dir / "final_selected_features.txt").write_text(
        "\n".join(result.final_features),
        encoding="utf-8",
    )

    _save_bar_plot(
        result.global_sage,
        "feature",
        "macro_sage_importance",
        sage_dir / "macro_sage_importance.png",
        "Class-balanced Macro-SAGE Importance",
        config.plot_top_n,
    )
    _save_bar_plot(
        result.global_shap,
        "feature",
        "mean_abs_shap",
        shap_dir / "global_shap_importance.png",
        "Global SHAP Importance",
        config.plot_top_n,
    )
    _save_bar_plot(
        result.global_cpf,
        "feature",
        "mean_classwise_cpf",
        cpf_dir / "global_cpf_importance.png",
        "Mean Class-wise Conditional Permutation Importance",
        config.plot_top_n,
    )
    if not result.interactions.empty:
        global_pairs = (
            result.interactions.assign(
                pair=lambda frame: frame["feature_1"] + " × " + frame["feature_2"]
            )
            .groupby("pair", as_index=False)["mean_abs_interaction"]
            .mean()
        )
        _save_bar_plot(
            global_pairs,
            "pair",
            "mean_abs_interaction",
            interaction_dir / "global_shap_interactions.png",
            "Mean SHAP Interaction Strength",
            config.plot_top_n,
        )

    with open(output_dir / "xai_config.json", "w", encoding="utf-8") as file:
        json.dump(
            asdict(config),
            file,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )

    readme = """XAI report layout
=================
sage/
  macro_sage_importance.csv
  classwise_sage_importance.csv
shap/
  global_shap_importance.csv
  classwise_shap_importance.csv
cpf/
  global_cpf_importance.csv
  classwise_cpf_importance.csv
interaction/
  classwise_shap_interactions.csv
lime/
  lime_local_explanations.csv
  classwise_lime_importance.csv
selection/
  classwise_feature_selection_scores.csv
  classwise_selected_features.csv
  final_selected_features.csv

Selection rule
--------------
1. Select the class-balanced Macro-SAGE top-k as the core feature set.
2. Add class protection from the intersection of class-wise SHAP top-k and
   positive class-wise CPFI top-k, with consensus fallback where necessary.
3. Fill to min_final_features in Macro-SAGE rank order.
4. Protect missing partners from valid class-wise SHAP interaction pairs up to
   max_final_features. Unsupported multiclass interaction output is skipped.
5. LIME is optional and is not part of automatic feature selection by default.

SAGE note
---------
All training rows participate in every repeat. Missing features use a random
donor permutation of the same training set. Per-class one-vs-rest binary
cross-entropy contributions are balanced 50/50 between positive and negative
rows, then averaged equally across classes to obtain Macro-SAGE.

CPFI note
---------
The CPFI implementation is an approximation: each feature is shuffled inside
quantile bins formed by its strongest correlated features. It does not sample
from the exact full conditional distribution p(X_j | X_-j).
"""
    (output_dir / "README.txt").write_text(readme, encoding="utf-8")


def run_xai_analysis(
    model: Any,
    X_reference: pd.DataFrame,
    y_reference: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    class_names: Sequence[str],
    output_dir: str | Path,
    config: XAIConfig | None = None,
) -> XAIResult:
    """Run full-train SAGE-led feature selection.

    X_reference and y_reference must be the complete training split used to fit
    the model. X_eval and y_eval are used only for optional LIME error analysis;
    they should be validation data, never the final test set.
    """
    config = config or XAIConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _ensure_numeric_frame(X_reference, "X_reference")
    _ensure_numeric_frame(X_eval, "X_eval")
    if list(X_reference.columns) != list(X_eval.columns):
        raise ValueError("X_reference and X_eval must have identical feature columns")

    class_ids = list(getattr(model, "classes_", sorted(pd.unique(y_reference))))
    if len(class_ids) != len(class_names):
        raise ValueError(
            "class_names must follow model.classes_ and have the same length. "
            f"classes={class_ids}, class_names={list(class_names)}"
        )

    global_sage, classwise_sage = compute_classwise_sage_importance(
        model=model,
        X=X_reference,
        y=y_reference,
        class_ids=class_ids,
        class_names=class_names,
        repeats=config.sage_repeats,
        random_state=config.random_state,
    )

    global_shap, classwise_shap, explainer = compute_classwise_shap_importance(
        model=model,
        X=X_reference,
        y=y_reference,
        class_ids=class_ids,
        class_names=class_names,
        batch_size=config.shap_batch_size,
    )

    global_cpf, classwise_cpf = compute_classwise_conditional_permutation_importance(
        model=model,
        X_reference=X_reference,
        X_eval=X_reference,
        y_eval=y_reference,
        class_ids=class_ids,
        class_names=class_names,
        config=config,
    )

    if config.enable_interaction:
        interactions = compute_classwise_shap_interactions(
            explainer=explainer,
            X=X_reference,
            y=y_reference,
            class_ids=class_ids,
            class_names=class_names,
            batch_size=config.interaction_batch_size,
        )
    else:
        interactions = pd.DataFrame()

    if config.enable_lime:
        lime_local, lime_aggregate = compute_lime_reports(
            model=model,
            X_reference=X_reference,
            X_eval=X_eval,
            y_eval=y_eval,
            class_ids=class_ids,
            class_names=class_names,
            config=config,
        )
    else:
        lime_local = pd.DataFrame()
        lime_aggregate = pd.DataFrame()

    selection_scores, classwise_selected, final_features = build_consensus_selection(
        feature_order=list(X_reference.columns),
        class_ids=class_ids,
        class_names=class_names,
        global_sage=global_sage,
        classwise_sage=classwise_sage,
        classwise_shap=classwise_shap,
        classwise_cpf=classwise_cpf,
        interactions=interactions,
        lime_aggregate=lime_aggregate,
        config=config,
    )

    result = XAIResult(
        global_sage=global_sage,
        classwise_sage=classwise_sage,
        global_shap=global_shap,
        classwise_shap=classwise_shap,
        global_cpf=global_cpf,
        classwise_cpf=classwise_cpf,
        interactions=interactions,
        lime_local=lime_local,
        lime_aggregate=lime_aggregate,
        selection_scores=selection_scores,
        classwise_selected=classwise_selected,
        final_features=final_features,
    )
    save_xai_reports(result, output_path, config)

    print("\n========== XAI Feature Selection ==========")
    print(f"Selected {len(final_features)} / {X_reference.shape[1]} features")
    print(final_features)
    print(f"[INFO] XAI reports saved to: {output_path}")
    return result
