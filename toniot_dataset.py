import os
import shutil
from pathlib import Path
import pandas as pd

import kagglehub

import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

DATASET_HANDLE = "arnobbhowmik/ton-iot-network-dataset"
DATA_PATH = Path("data_network/raw/train_test_network.csv")

'''
    데이터셋 다운로드하는 코드
'''

def make_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_dataset(dataset_handle: str = DATASET_HANDLE) -> Path:
    print(f"[INFO] Downloading dataset: {dataset_handle}")
    dataset_path = kagglehub.dataset_download(dataset_handle)
    print(f"[INFO] Download complete: {dataset_path}")
    return Path(dataset_path)


def copy_dataset(src_path: Path, dst_dir: str | Path) -> Path:
    dst_dir = make_dir(dst_dir)

    if src_path.is_file():
        target = dst_dir / src_path.name
        shutil.copy2(src_path, target)
        print(f"[INFO] Dataset copied to: {dst_dir}")
        return dst_dir

    for item in src_path.iterdir():
        target = dst_dir / item.name

        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    print(f"[INFO] Dataset copied to: {dst_dir}")
    return dst_dir


def list_files(dataset_dir: str | Path) -> None:
    dataset_dir = Path(dataset_dir)

    print("\n[INFO] Dataset files:")
    for path in dataset_dir.rglob("*"):
        if path.is_file():
            print(f" - {path}")


def download_toniot():
    save_dir = Path("data_network/raw")

    downloaded_path = download_dataset(DATASET_HANDLE)
    final_path = copy_dataset(downloaded_path, save_dir)
    list_files(final_path)


def ensure_raw_dataset():
    if DATA_PATH.exists():
        print(f"[INFO] Raw dataset already exists: {DATA_PATH}")
        return DATA_PATH

    print("[INFO] Raw dataset not found. Downloading raw dataset...")
    download_toniot()

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset download finished but expected file still missing: {DATA_PATH}"
        )

    return DATA_PATH

'''
    데이터셋 분석하는 코드
'''

def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"[INFO] Dataset loaded: {csv_path}")
    return df


def print_basic_info(df: pd.DataFrame) -> None:
    print("\n========== Basic Info ==========")
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns):,}")

    print("\nColumns:")
    for col in df.columns:
        print(f" - {col}")


def print_label_distribution(df: pd.DataFrame) -> None:
    print("\n========== Label Distribution ==========")

    if "label" not in df.columns:
        print("[WARN] 'label' column not found.")
        return

    label_counts = df["label"].value_counts(dropna=False)
    label_ratio = df["label"].value_counts(normalize=True, dropna=False) * 100

    result = pd.DataFrame({
        "count": label_counts,
        "ratio(%)": label_ratio.round(4)
    })

    print(result)


def print_type_distribution(df: pd.DataFrame) -> None:
    print("\n========== Attack Type Distribution ==========")

    if "type" not in df.columns:
        print("[WARN] 'type' column not found.")
        return

    type_counts = df["type"].value_counts(dropna=False)
    type_ratio = df["type"].value_counts(normalize=True, dropna=False) * 100

    result = pd.DataFrame({
        "count": type_counts,
        "ratio(%)": type_ratio.round(4)
    })

    print(result)

    print("\nAttack types:")
    for attack_type in type_counts.index:
        print(f" - {attack_type}")


def print_protocol_distribution(df: pd.DataFrame) -> None:
    print("\n========== Protocol Distribution ==========")

    if "proto" not in df.columns:
        print("[WARN] 'proto' column not found.")
        return

    print(df["proto"].value_counts(dropna=False))


def print_service_distribution(df: pd.DataFrame) -> None:
    print("\n========== Service Distribution ==========")

    if "service" not in df.columns:
        print("[WARN] 'service' column not found.")
        return

    print(df["service"].value_counts(dropna=False))


def print_connection_state_distribution(df: pd.DataFrame) -> None:
    print("\n========== Connection State Distribution ==========")

    if "conn_state" not in df.columns:
        print("[WARN] 'conn_state' column not found.")
        return

    print(df["conn_state"].value_counts(dropna=False))


def print_missing_values(df: pd.DataFrame) -> None:
    print("\n========== Missing / '-' Values ==========")

    missing_count = df.isna().sum()
    dash_count = (df == "-").sum()

    result = pd.DataFrame({
        "missing_count": missing_count,
        "dash_count": dash_count,
        "missing_ratio(%)": (missing_count / len(df) * 100).round(4),
        "dash_ratio(%)": (dash_count / len(df) * 100).round(4),
    })

    result = result[(result["missing_count"] > 0) | (result["dash_count"] > 0)]
    print(result)


def print_numeric_summary(df: pd.DataFrame) -> None:
    print("\n========== Numeric Feature Summary ==========")

    numeric_df = df.select_dtypes(include=["int64", "float64"])

    if numeric_df.empty:
        print("[WARN] No numeric columns found.")
        return

    print(numeric_df.describe().T)


def save_distribution_reports(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if "label" in df.columns:
        df["label"].value_counts(dropna=False).to_csv(
            output_dir / "label_distribution.csv",
            header=["count"]
        )

    if "type" in df.columns:
        df["type"].value_counts(dropna=False).to_csv(
            output_dir / "type_distribution.csv",
            header=["count"]
        )

    if "proto" in df.columns:
        df["proto"].value_counts(dropna=False).to_csv(
            output_dir / "proto_distribution.csv",
            header=["count"]
        )

    if "service" in df.columns:
        df["service"].value_counts(dropna=False).to_csv(
            output_dir / "service_distribution.csv",
            header=["count"]
        )

    print(f"\n[INFO] Reports saved to: {output_dir}")


def analyze_dataset(csv_path: Path, output_dir: Path) -> None:
    df = load_dataset(csv_path)

    print_basic_info(df)
    print_label_distribution(df)
    print_type_distribution(df)
    print_protocol_distribution(df)
    print_service_distribution(df)
    print_connection_state_distribution(df)
    print_missing_values(df)
    print_numeric_summary(df)

    save_distribution_reports(df, output_dir)


def analyze_toniot():
    output_dir = Path("reports/ton_iot_analysis")
    analyze_dataset(DATA_PATH, output_dir)

'''
    데이터셋 전처리
'''

DATA_PATH = Path("data_network/raw/train_test_network.csv")
OUTPUT_DIR = Path("data_network/processed_type")

TARGET_COLUMN = "type"

DROP_COLUMNS = [
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "label",  # binary classification에서는 label만 사용
]

DASH_DROP_THRESHOLD = 0.95
RANDOM_STATE = 42
VALID_SIZE = 0.15
TEST_SIZE = 0.15


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    print(f"[INFO] Loaded dataset: {path}")
    print(f"[INFO] Shape: {df.shape}")
    return df


def shuffle_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    print("[INFO] Dataset shuffled")
    return df


def drop_basic_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [col for col in DROP_COLUMNS if col in df.columns]
    df = df.drop(columns=drop_cols)

    print(f"[INFO] Dropped basic columns: {drop_cols}")
    return df


def drop_dash_heavy_columns(df: pd.DataFrame) -> pd.DataFrame:
    dash_ratio = (df == "-").sum() / len(df)

    drop_cols = dash_ratio[dash_ratio >= DASH_DROP_THRESHOLD].index.tolist()

    # target은 절대 제거하지 않음
    drop_cols = [col for col in drop_cols if col != TARGET_COLUMN]

    df = df.drop(columns=drop_cols)

    print(f"[INFO] Dropped dash-heavy columns threshold >= {DASH_DROP_THRESHOLD}")
    print(f"[INFO] Dropped columns: {drop_cols}")
    return df


def replace_dash_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace("-", "unknown")
    print("[INFO] Replaced '-' with 'unknown'")
    return df


def split_features_target(df: pd.DataFrame):
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column not found: {TARGET_COLUMN}")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    print(f"[INFO] X shape: {X.shape}")
    print(f"[INFO] y shape: {y.shape}")
    return X, y

def encode_target(
    y_train: pd.Series,
    y_valid: pd.Series,
    y_test: pd.Series,
):
    target_encoder = LabelEncoder()

    y_train_encoded = target_encoder.fit_transform(y_train)
    y_valid_encoded = target_encoder.transform(y_valid)
    y_test_encoded = target_encoder.transform(y_test)

    print("[INFO] Target classes:")
    for idx, class_name in enumerate(target_encoder.classes_):
        print(f"  {idx}: {class_name}")

    return (
        pd.Series(y_train_encoded, name=TARGET_COLUMN),
        pd.Series(y_valid_encoded, name=TARGET_COLUMN),
        pd.Series(y_test_encoded, name=TARGET_COLUMN),
        target_encoder,
    )

def encode_categorical_features(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
):
    encoders = {}

    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    print(f"[INFO] Categorical columns: {categorical_cols}")

    for col in categorical_cols:
        encoder = LabelEncoder()

        X_train[col] = X_train[col].astype(str)
        X_valid[col] = X_valid[col].astype(str)
        X_test[col] = X_test[col].astype(str)

        encoder.fit(X_train[col])

        X_train[col] = encoder.transform(X_train[col])

        # test에 train에 없던 값이 나오면 unknown 처리
        known_classes = set(encoder.classes_)
        X_valid[col] = X_valid[col].apply(
            lambda x: x if x in known_classes else "unknown"
        )
        X_test[col] = X_test[col].apply(
            lambda x: x if x in known_classes else "unknown"
        )

        if "unknown" not in encoder.classes_:
            encoder.classes_ = pd.Index(list(encoder.classes_) + ["unknown"]).to_numpy()

        X_valid[col] = encoder.transform(X_valid[col])
        X_test[col] = encoder.transform(X_test[col])

        encoders[col] = encoder

    return X_train, X_valid, X_test, encoders


def scale_features(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
):
    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns
    )

    X_valid_scaled = pd.DataFrame(
        X_valid_scaled,
        columns=X_valid.columns
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns
    )

    print("[INFO] MinMaxScaler applied")
    return X_train_scaled, X_valid_scaled, X_test_scaled, scaler


def save_outputs(
    X_train,
    X_valid,
    X_test,
    y_train,
    y_valid,
    y_test,
    encoders,
    target_encoder,
    scaler,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_valid.to_csv(output_dir / "X_valid.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_valid.to_csv(output_dir / "y_valid.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    joblib.dump(encoders, output_dir / "label_encoders.pkl")
    joblib.dump(scaler, output_dir / "minmax_scaler.pkl")
    joblib.dump(target_encoder, output_dir / "target_encoder.pkl")

    print(f"[INFO] Saved processed datasets to: {output_dir}")


def preprocess_dataset(output_dir: Path = OUTPUT_DIR):
    df = load_dataset(DATA_PATH)

    df = shuffle_dataset(df)
    df = drop_basic_columns(df)
    df = drop_dash_heavy_columns(df)
    df = replace_dash_values(df)

    X, y = split_features_target(df)

    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    valid_fraction_of_remainder = VALID_SIZE / (1 - TEST_SIZE)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=valid_fraction_of_remainder,
        random_state=RANDOM_STATE,
        stratify=y_train_valid,
    )

    print(
        "[INFO] Dataset split: "
        f"train={len(X_train)}, valid={len(X_valid)}, test={len(X_test)}"
    )

    y_train, y_valid, y_test, target_encoder = encode_target(
        y_train,
        y_valid,
        y_test,
    )

    X_train, X_valid, X_test, encoders = encode_categorical_features(
        X_train,
        X_valid,
        X_test,
    )
    X_train, X_valid, X_test, scaler = scale_features(
        X_train,
        X_valid,
        X_test,
    )

    save_outputs(
        X_train,
        X_valid,
        X_test,
        y_train.reset_index(drop=True),
        y_valid.reset_index(drop=True),
        y_test.reset_index(drop=True),
        encoders,
        target_encoder,
        scaler,
        output_dir,
    )


def ensure_processed_dataset(processed_dir: Path = OUTPUT_DIR):
    expected_files = [
        processed_dir / "X_train.csv",
        processed_dir / "X_valid.csv",
        processed_dir / "X_test.csv",
        processed_dir / "y_train.csv",
        processed_dir / "y_valid.csv",
        processed_dir / "y_test.csv",
    ]

    if all(path.exists() for path in expected_files):
        print(f"[INFO] Processed dataset already exists: {processed_dir}")
        return processed_dir

    print(f"[INFO] Processed dataset missing at {processed_dir}. Rebuilding from raw data...")
    ensure_raw_dataset()
    preprocess_dataset(output_dir=processed_dir)

    if not all(path.exists() for path in expected_files):
        raise FileNotFoundError(
            f"Processed dataset was not created properly in: {processed_dir}"
        )

    return processed_dir
