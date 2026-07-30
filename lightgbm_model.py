import lightgbm as lgb
from pathlib import Path
import joblib

def create_lightgbm_model(num_classes: int, random_state: int = 42) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        boosting_type="gbdt",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
        verbose=-1,
    )
    return model

def train_model(model, X_train, y_train):
    print("[INFO] Training LightGBM model...")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train)],
        eval_metric="multi_logloss",
        callbacks=[lgb.log_evaluation(period=20)],
    )

    print("[INFO] Training complete")
    return model

def save_model(model, model_dir: Path):
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "lightgbm_toniot_classification.pkl"

    joblib.dump(model, model_path)

    # model size
    model_size_bytes = model_path.stat().st_size
    model_size_kb = model_size_bytes / 1024
    model_size_mb = model_size_kb / 1024

    print(f"[INFO] Model saved to: {model_path}")
    print(f"[INFO] Model size: {model_size_kb:.2f} KB ({model_size_mb:.2f} MB)")