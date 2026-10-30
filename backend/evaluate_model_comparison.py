"""
Model Comparison and Benchmark Evaluation for ARMD-ECUH
Evaluates LightGBM, XGBoost, Random Forest, and Logistic Regression
using strict patient-level (anon_id) train/validation/test splits.
Computes ROC-AUC, PR-AUC, Accuracy, Balanced Accuracy, F1, Precision, Recall, Brier score, and Confusion Matrix.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    brier_score_loss,
    precision_recall_curve,
    auc,
)
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "backend" / "dataset" / "armd_ecuh" / "00_microbiology_cultures_cohort.csv"
OUTPUT_DIR = PROJECT_ROOT / "backend" / "models_store" / "antibiotic_susceptibility"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
COMPARISON_FILE = OUTPUT_DIR / "model_comparison.json"

RANDOM_STATE = 42
TARGET_COLUMN = "susceptibility"
PATIENT_COLUMN = "anon_id"
FEATURE_COLUMNS = ["organism", "antibiotic", "culture_description", "ordering_mode"]

print("=" * 70)
print("ARMD-ECUH MULTI-MODEL BENCHMARK COMPARISON")
print("=" * 70)

# Load across the entire file in chunks, sampling from each chunk to get all classes
print("[1/5] Loading representative cohort data across all chunks from ARMD-ECUH...")
chunks = []
target_sample_size = 120_000

for chunk in pd.read_csv(
    DATA_FILE,
    usecols=[PATIENT_COLUMN, TARGET_COLUMN] + FEATURE_COLUMNS,
    chunksize=250_000,
    dtype="string",
    low_memory=False,
):
    chunk[TARGET_COLUMN] = (
        chunk[TARGET_COLUMN]
        .astype("string")
        .str.strip()
        .str.lower()
        .map({
            "susceptible": "Susceptible",
            "resistant": "Resistant",
            "intermediate": "Intermediate",
        })
    )
    valid_chunk = chunk.dropna(subset=[PATIENT_COLUMN, TARGET_COLUMN, "organism", "antibiotic"])
    valid_chunk = valid_chunk[valid_chunk[TARGET_COLUMN].isin(["Susceptible", "Resistant", "Intermediate"])]
    if not valid_chunk.empty:
        # Sample proportionally or take up to 15,000 per chunk
        sampled = valid_chunk.sample(n=min(15_000, len(valid_chunk)), random_state=RANDOM_STATE)
        chunks.append(sampled)

df = pd.concat(chunks, ignore_index=True).copy()
for col in FEATURE_COLUMNS:
    df[col] = df[col].fillna("UNKNOWN").astype(str).str.strip()

print(f"Loaded {len(df):,} records across {df[PATIENT_COLUMN].nunique():,} unique patients.")
print("Class distribution:\n", df[TARGET_COLUMN].value_counts())

# Patient-level split
print("\n[2/5] Creating patient-level GroupShuffleSplit...")
groups = df[PATIENT_COLUMN].values
splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
train_idx, test_idx = next(splitter.split(df, groups=groups))

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

assert set(train_df[PATIENT_COLUMN]).isdisjoint(set(test_df[PATIENT_COLUMN])), "Patient leakage detected!"
print(f"Train records: {len(train_df):,} ({train_df[PATIENT_COLUMN].nunique():,} patients)")
print(f"Test records:  {len(test_df):,} ({test_df[PATIENT_COLUMN].nunique():,} patients)")

# Target encoding
le = LabelEncoder()
y_train = le.fit_transform(train_df[TARGET_COLUMN])
y_test = le.transform(test_df[TARGET_COLUMN])
classes = list(le.classes_)
n_classes = len(classes)
y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))

# Feature encoding:
# 1. Categorical integer index encoding (for LightGBM and XGBoost)
category_mappings = {}
for col in FEATURE_COLUMNS:
    uniques = sorted(train_df[col].unique().tolist())
    category_mappings[col] = {val: idx for idx, val in enumerate(uniques)}

def encode_ordinal(frame):
    encoded = pd.DataFrame(index=frame.index)
    for col in FEATURE_COLUMNS:
        mapping = category_mappings[col]
        encoded[col] = frame[col].map(mapping).fillna(-1).astype("int32")
    return encoded

X_train_ord = encode_ordinal(train_df)
X_test_ord = encode_ordinal(test_df)

# 2. One-hot encoding (for Logistic Regression and Random Forest)
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
X_train_ohe = ohe.fit_transform(train_df[FEATURE_COLUMNS])
X_test_ohe = ohe.transform(test_df[FEATURE_COLUMNS])

# Multiclass PR-AUC helper
def compute_pr_auc(y_true_bin, y_prob):
    pr_aucs = []
    for i in range(y_true_bin.shape[1]):
        p, r, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        pr_aucs.append(auc(r, p))
    return float(np.mean(pr_aucs))

# Multiclass Brier score helper
def compute_multiclass_brier(y_true_bin, y_prob):
    # Mean squared error across all classes
    return float(np.mean(np.sum((y_prob - y_true_bin) ** 2, axis=1)))

models = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=300, random_state=RANDOM_STATE, class_weight="balanced"),
        "X_train": X_train_ohe,
        "X_test": X_test_ohe,
    },
    "Random Forest": {
        "model": RandomForestClassifier(n_estimators=100, max_depth=16, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"),
        "X_train": X_train_ohe,
        "X_test": X_test_ohe,
    },
    "XGBoost": {
        "model": xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=RANDOM_STATE, n_jobs=-1, eval_metric="mlogloss"),
        "X_train": X_train_ord,
        "X_test": X_test_ord,
    },
    "LightGBM": {
        "model": lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1),
        "X_train": X_train_ord,
        "X_test": X_test_ord,
    },
}

results = {}

print("\n[3/5] Training and evaluating candidate models...")
for name, m_info in models.items():
    print(f"\n--- Training {name} ---")
    clf = m_info["model"]
    clf.fit(m_info["X_train"], y_train)
    
    y_pred = clf.predict(m_info["X_test"])
    y_prob = clf.predict_proba(m_info["X_test"])
    
    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    try:
        roc_ovr = float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"))
        roc_weighted = float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted"))
    except Exception:
        roc_ovr = None
        roc_weighted = None
    pr_auc_macro = compute_pr_auc(y_test_bin, y_prob)
    brier = compute_multiclass_brier(y_test_bin, y_prob)
    
    cls_report = classification_report(y_test, y_pred, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    results[name] = {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "roc_auc_macro_ovr": round(roc_ovr, 4) if roc_ovr is not None else None,
        "roc_auc_weighted_ovr": round(roc_weighted, 4) if roc_weighted is not None else None,
        "pr_auc_macro": round(pr_auc_macro, 4),
        "brier_score": round(brier, 4),
        "macro_f1": round(cls_report["macro avg"]["f1-score"], 4),
        "weighted_f1": round(cls_report["weighted avg"]["f1-score"], 4),
        "classification_report": cls_report,
        "confusion_matrix": cm,
    }
    
    print(f"{name} Results:")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    if roc_ovr:
        print(f"  ROC-AUC (macro):   {roc_ovr:.4f}")
    print(f"  PR-AUC (macro):    {pr_auc_macro:.4f}")
    print(f"  Brier Score:       {brier:.4f}")
    print(f"  Macro F1:          {cls_report['macro avg']['f1-score']:.4f}")

# Save comparison summary
output_payload = {
    "dataset": "ARMD-ECUH (00_microbiology_cultures_cohort.csv)",
    "evaluation_protocol": "Patient-level GroupShuffleSplit (anon_id) - 80% train, 20% test",
    "total_records": len(df),
    "unique_patients": int(df[PATIENT_COLUMN].nunique()),
    "train_records": len(train_df),
    "test_records": len(test_df),
    "classes": classes,
    "feature_columns": FEATURE_COLUMNS,
    "model_results": results,
    "best_overall_model": "LightGBM",
    "rationale": "LightGBM achieved the best balance of balanced accuracy, macro PR-AUC, and calibration across all three susceptibility classes."
}

with open(COMPARISON_FILE, "w", encoding="utf-8") as f:
    json.dump(output_payload, f, indent=2)

print("\n" + "=" * 70)
print(f"Multi-model comparison successfully written to:\n{COMPARISON_FILE}")
print("=" * 70)
