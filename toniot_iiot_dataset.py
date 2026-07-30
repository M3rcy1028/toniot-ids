from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# ============================================================
# UNSW TON_IoT Train_Test_IoT_dataset 전처리 코드
# ============================================================

RAW_DATA_DIR = Path("data")
REPORT_ROOT_DIR = Path("reports/ton_iot_analysis")
PROCESSED_ROOT_DIR = Path("data/processed_iot")

TARGET_COLUMN = "type"

DROP_COLUMNS = [
    "label",
    "date",
    "time",
    "timestamp",
    "ts",
]

DASH_DROP_THRESHOLD = 0.95
RANDOM_STATE = 42
TEST_SIZE = 0.3

EXPECTED_DATASET_FILES = [
    "Train_Test_IoT_Fridge.csv",
    "Train_Test_IoT_Garage_Door.csv",
    "Train_Test_IoT_GPS_Tracker.csv",
    "Train_Test_IoT_Modbus.csv",
    "Train_Test_IoT_Motion_Light.csv",
    "Train_Test_IoT_Thermostat.csv",
    "Train_Test_IoT_Weather.csv",
]


def make_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device_name(csv_path: str | Path) -> str:
    csv_path = Path(csv_path)
    return csv_path.stem.replace("Train_Test_IoT_", "").lower()


def find_dataset_files(data_dir: str | Path = RAW_DATA_DIR) -> list[Path]:
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    dataset_files = []

    for filename in EXPECTED_DATASET_FILES:
        csv_path = data_dir / filename

        if csv_path.exists():
            dataset_files.append(csv_path)
        else:
            print(f"[WARN] Expected dataset not found: {csv_path}")

    if not dataset_files:
        dataset_files = sorted(data_dir.glob("Train_Test_IoT_*.csv"))

    if not dataset_files:
        raise FileNotFoundError(
            f"No Train_Test_IoT_*.csv files found in: {data_dir}"
        )

    return dataset_files


def list_files(dataset_dir: str | Path = RAW_DATA_DIR) -> None:
    dataset_dir = Path(dataset_dir)

    print("\n[INFO] Dataset files:")
    for path in sorted(dataset_dir.glob("Train_Test_IoT_*.csv")):
        print(f" - {path}")


def ensure_raw_dataset(data_dir: str | Path = RAW_DATA_DIR) -> list[Path]:
    dataset_files = find_dataset_files(data_dir)
    print(f"[INFO] Found {len(dataset_files)} IoT dataset files")
    return dataset_files


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()

    print(f"[INFO] Dataset loaded: {csv_path}")
    print(f"[INFO] Shape: {df.shape}")
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
    label_ratio = (
        df["label"].value_counts(normalize=True, dropna=False) * 100
    )

    result = pd.DataFrame({
        "count": label_counts,
        "ratio(%)": label_ratio.round(4),
    })

    print(result)


def print_type_distribution(df: pd.DataFrame) -> None:
    print("\n========== Attack Type Distribution ==========")

    if TARGET_COLUMN not in df.columns:
        print(f"[WARN] '{TARGET_COLUMN}' column not found.")
        return

    type_counts = df[TARGET_COLUMN].value_counts(dropna=False)
    type_ratio = (
        df[TARGET_COLUMN].value_counts(normalize=True, dropna=False) * 100
    )

    result = pd.DataFrame({
        "count": type_counts,
        "ratio(%)": type_ratio.round(4),
    })

    print(result)

    print("\nAttack types:")
    for attack_type in type_counts.index:
        print(f" - {attack_type}")


def print_missing_values(df: pd.DataFrame) -> None:
    print("\n========== Missing / '-' Values ==========")

    missing_count = df.isna().sum()
    dash_count = pd.Series({
        col: df[col].astype(str).eq("-").sum()
        for col in df.columns
    })

    result = pd.DataFrame({
        "missing_count": missing_count,
        "dash_count": dash_count,
        "missing_ratio(%)": (missing_count / len(df) * 100).round(4),
        "dash_ratio(%)": (dash_count / len(df) * 100).round(4),
    })

    result = result[
        (result["missing_count"] > 0)
        | (result["dash_count"] > 0)
    ]

    if result.empty:
        print("[INFO] Missing values and '-' values were not found.")
    else:
        print(result)


def print_numeric_summary(df: pd.DataFrame) -> None:
    print("\n========== Numeric Feature Summary ==========")

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        print("[WARN] No numeric columns found.")
        return

    print(numeric_df.describe().T)


def save_distribution_reports(
    df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    output_dir = make_dir(output_dir)

    if "label" in df.columns:
        label_counts = df["label"].value_counts(dropna=False)
        label_ratio = (
            df["label"].value_counts(normalize=True, dropna=False) * 100
        )

        pd.DataFrame({
            "count": label_counts,
            "ratio(%)": label_ratio.round(4),
        }).to_csv(output_dir / "label_distribution.csv")

    if TARGET_COLUMN in df.columns:
        type_counts = df[TARGET_COLUMN].value_counts(dropna=False)
        type_ratio = (
            df[TARGET_COLUMN].value_counts(normalize=True, dropna=False) * 100
        )

        pd.DataFrame({
            "count": type_counts,
            "ratio(%)": type_ratio.round(4),
        }).to_csv(output_dir / "type_distribution.csv")

    missing_count = df.isna().sum()
    dash_count = pd.Series({
        col: df[col].astype(str).eq("-").sum()
        for col in df.columns
    })

    pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "unique_count": df.nunique(dropna=True),
        "missing_count": missing_count,
        "dash_count": dash_count,
        "missing_ratio(%)": (missing_count / len(df) * 100).round(4),
        "dash_ratio(%)": (dash_count / len(df) * 100).round(4),
    }).to_csv(output_dir / "column_summary.csv")

    numeric_df = df.select_dtypes(include="number")

    if not numeric_df.empty:
        numeric_df.describe().T.to_csv(
            output_dir / "numeric_summary.csv"
        )

    print(f"\n[INFO] Reports saved to: {output_dir}")


def analyze_dataset(
    csv_path: str | Path,
    output_dir: str | Path,
) -> None:
    df = load_dataset(csv_path)

    print_basic_info(df)
    print_label_distribution(df)
    print_type_distribution(df)
    print_missing_values(df)
    print_numeric_summary(df)

    save_distribution_reports(df, output_dir)


def analyze_all_toniot_devices() -> None:
    dataset_files = ensure_raw_dataset()

    for csv_path in dataset_files:
        device_name = get_device_name(csv_path)
        output_dir = REPORT_ROOT_DIR / device_name

        print("\n" + "=" * 70)
        print(f"[INFO] Analyzing device: {device_name}")
        print("=" * 70)

        analyze_dataset(csv_path, output_dir)


def shuffle_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    print("[INFO] Dataset shuffled")
    return df


def drop_basic_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        col for col in DROP_COLUMNS
        if col in df.columns
    ]

    df = df.drop(columns=drop_cols)

    print(f"[INFO] Dropped basic columns: {drop_cols}")
    return df


def drop_dash_heavy_columns(df: pd.DataFrame) -> pd.DataFrame:
    dash_ratio = pd.Series({
        col: df[col].astype(str).eq("-").mean()
        for col in df.columns
    })

    drop_cols = dash_ratio[
        dash_ratio >= DASH_DROP_THRESHOLD
    ].index.tolist()

    drop_cols = [
        col for col in drop_cols
        if col != TARGET_COLUMN
    ]

    df = df.drop(columns=drop_cols)

    print(
        "[INFO] Dropped dash-heavy columns "
        f"threshold >= {DASH_DROP_THRESHOLD}"
    )
    print(f"[INFO] Dropped columns: {drop_cols}")

    return df


def replace_dash_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace("-", "unknown")
    print("[INFO] Replaced '-' with 'unknown'")
    return df


def drop_missing_target_rows(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column not found: {TARGET_COLUMN}")

    before = len(df)
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    after = len(df)

    if before != after:
        print(
            f"[INFO] Dropped {before - after} rows "
            "with missing target values"
        )

    return df


def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    constant_cols = [
        col for col in df.columns
        if col != TARGET_COLUMN
        and df[col].nunique(dropna=False) <= 1
    ]

    df = df.drop(columns=constant_cols)

    print(f"[INFO] Dropped constant columns: {constant_cols}")
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [
        col for col in numeric_cols
        if col != TARGET_COLUMN
    ]

    categorical_cols = [
        col for col in df.columns
        if col not in numeric_cols
        and col != TARGET_COLUMN
    ]

    for col in numeric_cols:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    for col in categorical_cols:
        df[col] = df[col].fillna("unknown")

    print("[INFO] Missing values filled")
    return df


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column not found: {TARGET_COLUMN}")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(str)

    print(f"[INFO] X shape: {X.shape}")
    print(f"[INFO] y shape: {y.shape}")

    return X, y


def encode_target(
    y_train: pd.Series,
    y_test: pd.Series,
):
    target_encoder = LabelEncoder()

    y_train_encoded = target_encoder.fit_transform(y_train)

    unknown_test_classes = sorted(
        set(y_test.unique())
        - set(target_encoder.classes_)
    )

    if unknown_test_classes:
        raise ValueError(
            "Test target contains classes that do not exist "
            f"in training data: {unknown_test_classes}"
        )

    y_test_encoded = target_encoder.transform(y_test)

    print("[INFO] Target classes:")

    for idx, class_name in enumerate(target_encoder.classes_):
        print(f"  {idx}: {class_name}")

    return (
        pd.Series(y_train_encoded, name=TARGET_COLUMN),
        pd.Series(y_test_encoded, name=TARGET_COLUMN),
        target_encoder,
    )


def encode_categorical_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
):
    X_train = X_train.copy()
    X_test = X_test.copy()

    encoders = {}

    categorical_cols = X_train.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    print(f"[INFO] Categorical columns: {categorical_cols}")

    for col in categorical_cols:
        encoder = LabelEncoder()

        X_train[col] = (
            X_train[col]
            .fillna("unknown")
            .astype(str)
        )

        X_test[col] = (
            X_test[col]
            .fillna("unknown")
            .astype(str)
        )

        train_values = X_train[col].tolist()

        if "unknown" not in train_values:
            train_values.append("unknown")

        encoder.fit(train_values)

        known_classes = set(encoder.classes_)

        X_train[col] = X_train[col].apply(
            lambda value: value if value in known_classes else "unknown"
        )

        X_test[col] = X_test[col].apply(
            lambda value: value if value in known_classes else "unknown"
        )

        X_train[col] = encoder.transform(X_train[col])
        X_test[col] = encoder.transform(X_test[col])

        encoders[col] = encoder

    return X_train, X_test, encoders


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
):
    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns,
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns,
    )

    print("[INFO] MinMaxScaler applied")
    return X_train_scaled, X_test_scaled, scaler


def save_outputs(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    encoders: dict,
    target_encoder: LabelEncoder,
    scaler: MinMaxScaler,
    output_dir: str | Path,
) -> None:
    output_dir = make_dir(output_dir)

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    joblib.dump(encoders, output_dir / "label_encoders.pkl")
    joblib.dump(scaler, output_dir / "minmax_scaler.pkl")
    joblib.dump(target_encoder, output_dir / "target_encoder.pkl")

    class_mapping = pd.DataFrame({
        "class_id": range(len(target_encoder.classes_)),
        "class_name": target_encoder.classes_,
    })

    class_mapping.to_csv(
        output_dir / "class_mapping.csv",
        index=False,
    )

    print(f"[INFO] Saved processed datasets to: {output_dir}")


def preprocess_dataset(
    data_path: str | Path,
    output_dir: str | Path,
) -> dict:
    data_path = Path(data_path)
    output_dir = Path(output_dir)

    df = load_dataset(data_path)

    df = shuffle_dataset(df)
    df = drop_missing_target_rows(df)
    df = drop_basic_columns(df)
    df = drop_dash_heavy_columns(df)
    df = replace_dash_values(df)
    df = drop_constant_columns(df)
    df = fill_missing_values(df)

    X, y = split_features_target(df)

    class_counts = y.value_counts()

    if class_counts.min() < 2:
        print(
            "[WARN] A target class has fewer than 2 samples. "
            "Stratified split cannot be used."
        )
        stratify = None
    else:
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    y_train, y_test, target_encoder = encode_target(
        y_train,
        y_test,
    )

    X_train, X_test, encoders = encode_categorical_features(
        X_train,
        X_test,
    )

    X_train, X_test, scaler = scale_features(
        X_train,
        X_test,
    )

    save_outputs(
        X_train,
        X_test,
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
        encoders,
        target_encoder,
        scaler,
        output_dir,
    )

    print("\n[INFO] Processed data loaded")
    print(f"[INFO] X_train: {X_train.shape}")
    print(f"[INFO] X_test : {X_test.shape}")

    return {
        "source_file": data_path.name,
        "device": get_device_name(data_path),
        "rows": len(df),
        "features": X_train.shape[1],
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "classes": len(target_encoder.classes_),
        "output_dir": str(output_dir),
    }


def preprocess_all_devices() -> pd.DataFrame:
    dataset_files = ensure_raw_dataset()
    summary_rows = []

    for csv_path in dataset_files:
        device_name = get_device_name(csv_path)
        output_dir = PROCESSED_ROOT_DIR / device_name

        print("\n" + "=" * 70)
        print(f"[INFO] Preprocessing device: {device_name}")
        print("=" * 70)

        try:
            result = preprocess_dataset(
                csv_path,
                output_dir,
            )

            result["status"] = "success"
            result["error"] = ""

        except Exception as error:
            print(
                f"[ERROR] Failed to preprocess "
                f"{csv_path.name}: {error}"
            )

            result = {
                "source_file": csv_path.name,
                "device": device_name,
                "rows": "",
                "features": "",
                "train_samples": "",
                "test_samples": "",
                "classes": "",
                "output_dir": str(output_dir),
                "status": "failed",
                "error": str(error),
            }

        summary_rows.append(result)

    summary_df = pd.DataFrame(summary_rows)

    make_dir(REPORT_ROOT_DIR)

    summary_df.to_csv(
        REPORT_ROOT_DIR / "all_devices_summary.csv",
        index=False,
    )

    print("\n========== All Device Summary ==========")
    print(summary_df.to_string(index=False))

    return summary_df


def ensure_processed_dataset(
    processed_root_dir: str | Path = PROCESSED_ROOT_DIR,
) -> Path:
    processed_root_dir = Path(processed_root_dir)
    dataset_files = ensure_raw_dataset()

    rebuild_required = False

    for csv_path in dataset_files:
        device_name = get_device_name(csv_path)
        processed_dir = processed_root_dir / device_name

        expected_files = [
            processed_dir / "X_train.csv",
            processed_dir / "X_test.csv",
            processed_dir / "y_train.csv",
            processed_dir / "y_test.csv",
            processed_dir / "label_encoders.pkl",
            processed_dir / "minmax_scaler.pkl",
            processed_dir / "target_encoder.pkl",
        ]

        if not all(path.exists() for path in expected_files):
            rebuild_required = True
            break

    if not rebuild_required:
        print(
            "[INFO] Processed datasets already exist: "
            f"{processed_root_dir}"
        )
        return processed_root_dir

    print(
        "[INFO] One or more processed datasets are missing. "
        "Rebuilding all IoT device datasets..."
    )

    preprocess_all_devices()

    return processed_root_dir


def main() -> None:
    list_files(RAW_DATA_DIR)
    analyze_all_toniot_devices()
    preprocess_all_devices()


if __name__ == "__main__":
    main()
