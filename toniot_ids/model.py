import lightgbm as lgb
from pathlib import Path
import joblib

def create_lightgbm_model(num_classes: int, random_state: int = 42) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        boosting_type="gbdt",
        n_estimators=3000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        # Avoid oversubscription on the 80-logical-CPU experiment server.
        n_jobs=8,
        class_weight=None,
        verbose=-1,
    )
    return model

def train_model(model, X_train, y_train, X_valid=None, y_valid=None):
    print("[INFO] Training LightGBM model...")

    callbacks = [lgb.log_evaluation(period=20)]
    fit_kwargs = {}
    if X_valid is not None and y_valid is not None:
        fit_kwargs["eval_set"] = [(X_valid, y_valid)]
        fit_kwargs["eval_metric"] = "multi_logloss"
        callbacks.insert(0, lgb.early_stopping(stopping_rounds=50))

    model.fit(
        X_train,
        y_train,
        callbacks=callbacks,
        **fit_kwargs,
    )

    if getattr(model, "best_iteration_", None):
        print(f"[INFO] Best iteration: {model.best_iteration_}")
    print("[INFO] Training complete")
    return model

def save_model(
    model,
    model_dir: Path,
    filename: str = "lightgbm_toniot_classification.pkl",
):
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / filename

    joblib.dump(model, model_path)

    # model size
    model_size_bytes = model_path.stat().st_size
    model_size_kb = model_size_bytes / 1024
    model_size_mb = model_size_kb / 1024

    print(f"[INFO] Model saved to: {model_path}")
    print(f"[INFO] Model size: {model_size_kb:.2f} KB ({model_size_mb:.2f} MB)")
