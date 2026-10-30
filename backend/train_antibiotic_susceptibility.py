from pathlib import Path
import json
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------
# PATHS
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "backend"
    / "dataset"
    / "armd_ecuh"
    / "00_microbiology_cultures_cohort.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "backend"
    / "models_store"
    / "antibiotic_susceptibility"
)

MODEL_FILE = OUTPUT_DIR / "lightgbm_susceptibility.joblib"
METADATA_FILE = OUTPUT_DIR / "metadata.json"
METRICS_FILE = OUTPUT_DIR / "metrics.json"

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
REQUIRED_COLUMNS = [
    "anon_id",
    "organism",
    "antibiotic",
    "susceptibility",
]

OPTIONAL_FEATURES = [
    "culture_description",
    "ordering_mode",
]

TARGET_COLUMN = "susceptibility"
PATIENT_COLUMN = "anon_id"

RANDOM_STATE = 42


def clean_text(series):
    """Normalize missing values and text consistently."""
    result = series.astype("string").str.strip()

    missing_values = [
        "",
        "null",
        "none",
        "nan",
        "n/a",
    ]

    return result.mask(
        result.str.lower().isin(missing_values),
        pd.NA,
    )


def main():
    print("=" * 65)
    print("SEPSISGUARDIAN AI - ANTIBIOTIC SUSCEPTIBILITY TRAINING")
    print("=" * 65)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. READ ONLY THE REQUIRED COLUMNS
    # --------------------------------------------------
    print("\n[1/7] Inspecting dataset columns...")

    header = pd.read_csv(
        DATA_FILE,
        nrows=0,
    )

    available_columns = list(header.columns)

    missing_columns = [
        col
        for col in REQUIRED_COLUMNS
        if col not in available_columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns missing from dataset: "
            + ", ".join(missing_columns)
            + "\nAvailable columns: "
            + ", ".join(available_columns)
        )

    feature_columns = [
        "organism",
        "antibiotic",
    ]

    for col in OPTIONAL_FEATURES:
        if col in available_columns:
            feature_columns.append(col)

    columns_to_load = list(
        dict.fromkeys(
            REQUIRED_COLUMNS
            + feature_columns
        )
    )

    print("Features:", feature_columns)
    print("Loading selected columns...")

    df = pd.read_csv(
        DATA_FILE,
        usecols=columns_to_load,
        dtype="string",
        low_memory=False,
    )

    print(f"Rows loaded: {len(df):,}")

    # --------------------------------------------------
    # 2. CLEAN LABELS AND FEATURES
    # --------------------------------------------------
    print("\n[2/7] Cleaning data...")

    for col in columns_to_load:
        df[col] = clean_text(df[col])

    # This version trains a three-class model.
    valid_labels = [
        "Susceptible",
        "Resistant",
        "Intermediate",
    ]

    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .str.lower()
        .map({
            "susceptible": "Susceptible",
            "resistant": "Resistant",
            "intermediate": "Intermediate",
        })
    )

    # Exclude missing patient IDs so evaluation can
    # keep patients separated between data splits.
    df = df.dropna(
        subset=[
            PATIENT_COLUMN,
            TARGET_COLUMN,
            "organism",
            "antibiotic",
        ]
    ).copy()

    df = df[
        df[TARGET_COLUMN].isin(valid_labels)
    ].copy()

    if df.empty:
        raise ValueError(
            "No valid labeled records remain after cleaning."
        )

    print(f"Valid labeled records: {len(df):,}")
    print("\nLabel distribution:")
    print(df[TARGET_COLUMN].value_counts())

    present_labels = set(df[TARGET_COLUMN].unique())
    if present_labels != set(valid_labels):
        raise ValueError(
            "Expected all three target classes, but found: "
            f"{sorted(present_labels)}"
        )

    # Missing optional feature values become UNKNOWN.
    # This avoids the Categorical.fillna(-1) error.
    for col in feature_columns:
        df[col] = df[col].fillna("UNKNOWN").astype(str)

    # --------------------------------------------------
    # 3. PATIENT-LEVEL TRAIN / VALIDATION / TEST SPLIT
    # --------------------------------------------------
    print("\n[3/7] Creating patient-level data splits...")

    groups = df[PATIENT_COLUMN].astype(str).to_numpy()
    indices = np.arange(len(df))

    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    train_val_pos, test_pos = next(
        first_split.split(
            indices,
            groups=groups,
        )
    )

    train_val_df = df.iloc[train_val_pos].copy()
    test_df = df.iloc[test_pos].copy()

    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    train_pos, val_pos = next(
        second_split.split(
            np.arange(len(train_val_df)),
            groups=train_val_df[PATIENT_COLUMN].astype(str).to_numpy(),
        )
    )

    train_df = train_val_df.iloc[train_pos].copy()
    val_df = train_val_df.iloc[val_pos].copy()

    train_patients = set(train_df[PATIENT_COLUMN].astype(str))
    val_patients = set(val_df[PATIENT_COLUMN].astype(str))
    test_patients = set(test_df[PATIENT_COLUMN].astype(str))

    assert train_patients.isdisjoint(val_patients)
    assert train_patients.isdisjoint(test_patients)
    assert val_patients.isdisjoint(test_patients)

    print(f"Train records: {len(train_df):,}")
    print(f"Validation records: {len(val_df):,}")
    print(f"Test records: {len(test_df):,}")

    # --------------------------------------------------
    # 4. ENCODE CATEGORICAL FEATURES
    # --------------------------------------------------
    print("\n[4/7] Encoding categorical features...")

    # Mappings are learned only from the training data.
    category_mappings = {}

    for col in feature_columns:
        unique_values = sorted(
            train_df[col].astype(str).unique().tolist()
        )

        category_mappings[col] = {
            value: index
            for index, value in enumerate(unique_values)
        }

    def encode_features(frame):
        encoded = pd.DataFrame(index=frame.index)

        for col in feature_columns:
            mapping = category_mappings[col]

            # Unseen categories are mapped to -1.
            # No Categorical.fillna operation is used.
            encoded[col] = (
                frame[col]
                .astype(str)
                .map(mapping)
                .fillna(-1)
                .astype("int32")
            )

        return encoded

    X_train = encode_features(train_df)
    X_val = encode_features(val_df)
    X_test = encode_features(test_df)

    label_encoder = LabelEncoder()

    y_train = label_encoder.fit_transform(
        train_df[TARGET_COLUMN]
    )

    y_val = label_encoder.transform(
        val_df[TARGET_COLUMN]
    )

    y_test = label_encoder.transform(
        test_df[TARGET_COLUMN]
    )

    print("Target classes:", label_encoder.classes_.tolist())

    # --------------------------------------------------
    # 5. TRAIN LIGHTGBM
    # --------------------------------------------------
    print("\n[5/7] Training LightGBM model...")
    print("Please wait. Large datasets may take time.")

    model = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=len(label_encoder.classes_),
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        categorical_feature=feature_columns,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=30,
                verbose=True,
            ),
            lgb.log_evaluation(period=25),
        ],
    )

    # --------------------------------------------------
    # 6. EVALUATE ON HELD-OUT TEST PATIENTS
    # --------------------------------------------------
    print("\n[6/7] Evaluating on test patients...")

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions,
    )

    report = classification_report(
        y_test,
        predictions,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_.tolist(),
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=np.arange(len(label_encoder.classes_)),
    )

    print(f"\nTest accuracy: {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=np.arange(len(label_encoder.classes_)),
            target_names=label_encoder.classes_.tolist(),
            zero_division=0,
        )
    )

    print("Confusion matrix (true rows, predicted columns):")
    print(matrix)

    metrics = {
        "dataset": str(DATA_FILE),
        "target": TARGET_COLUMN,
        "feature_columns": feature_columns,
        "classes": label_encoder.classes_.tolist(),
        "split_method": "GroupShuffleSplit by anon_id",
        "train_records": int(len(train_df)),
        "validation_records": int(len(val_df)),
        "test_records": int(len(test_df)),
        "train_unique_patients": int(len(train_patients)),
        "validation_unique_patients": int(len(val_patients)),
        "test_unique_patients": int(len(test_patients)),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "best_iteration": int(
            model.best_iteration_
            if model.best_iteration_
            else model.n_estimators
        ),
        "note": (
            "Research prototype. Predicts culture susceptibility "
            "classes; does not independently establish the best "
            "antibiotic for an individual patient."
        ),
    }

    # --------------------------------------------------
    # 7. SAVE MODEL AND METADATA
    # --------------------------------------------------
    print("\n[7/7] Saving model files...")

    bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "category_mappings": category_mappings,
        "label_encoder": label_encoder,
    }

    joblib.dump(bundle, MODEL_FILE)

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "feature_columns": feature_columns,
                "category_mappings": category_mappings,
                "classes": label_encoder.classes_.tolist(),
                "target_column": TARGET_COLUMN,
                "patient_group_column": PATIENT_COLUMN,
            },
            file,
            indent=2,
        )

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    print("\n" + "=" * 65)
    print("TRAINING FINISHED")
    print("=" * 65)
    print(f"Model:   {MODEL_FILE}")
    print(f"Metadata: {METADATA_FILE}")
    print(f"Metrics: {METRICS_FILE}")
    print("\nNext: review the test metrics before integrating the API.")


if __name__ == "__main__":
    main()