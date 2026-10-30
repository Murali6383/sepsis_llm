"""
SepsisGuardian AI — Model Training Pipeline
============================================
Trains all six LightGBM classifiers and saves fitted pipelines to models_store/.

Usage (from backend/ directory):
    python -m app.models.train_models

Each pipeline = ColumnTransformer preprocessing + LightGBM classifier.
Preprocessing is fitted on the training split ONLY.

Datasets used (all synthetic — research demonstration only):
  - synthetic_sepsis_training.csv         → sepsis_label
  - synthetic_aki_training.csv            → aki_label
  - synthetic_cardiovascular_risk_training.csv → cardiovascular_risk_label
  - synthetic_liver_risk_training.csv     → liver_risk_label
  - synthetic_neurological_risk_training.csv   → neurological_risk_label
  - synthetic_respiratory_risk_training.csv    → respiratory_risk_label
  - synthetic_antibiotic.csv              → recommended_antibiotic_demo_label (multiclass)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("train_models")

# ── Paths ─────────────────────────────────────────────────────────────────────
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent  # backend/
_PROJECT_ROOT = _BACKEND_DIR.parent
_MODELS_DIR = _BACKEND_DIR / "models_store"
_MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_numeric_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def _make_categorical_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])


def _build_pipeline(num_features: list[str], cat_features: list[str]) -> Pipeline:
    transformers = []
    if num_features:
        transformers.append(("num", _make_numeric_pipeline(), num_features))
    if cat_features:
        transformers.append(("cat", _make_categorical_pipeline(), cat_features))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    clf = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", clf)])


def _split(df: pd.DataFrame, target: str, random_state: int = RANDOM_STATE):
    """Stratified split; keeps patient_id groups together."""
    groups = df["patient_id"] if "patient_id" in df.columns else None
    X = df.drop(columns=[target])
    y = df[target]
    if groups is not None:
        # Use GroupShuffleSplit to avoid patient overlap
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups))
        return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=random_state)


def _evaluate_binary(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, name: str) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()
    logger.info(f"\n{'='*50}\n{name}\n{'='*50}")
    logger.info(f"ROC-AUC : {roc:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    logger.info(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")
    return {"roc_auc": round(roc, 4), "classification_report": report, "confusion_matrix": cm}


def _evaluate_multiclass(model: Pipeline, X_test: pd.DataFrame, y_test, le: LabelEncoder, name: str) -> dict:
    y_pred_enc = model.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)
    y_true = le.inverse_transform(y_test)
    report = classification_report(y_true, y_pred, output_dict=True)
    logger.info(f"\n{'='*50}\n{name}\n{'='*50}")
    logger.info(f"\n{classification_report(y_true, y_pred)}")
    return {"classification_report": report}


# ── Individual Model Trainers ─────────────────────────────────────────────────

def train_sepsis():
    logger.info("Training: Sepsis model")
    df = pd.read_csv(_PROJECT_ROOT / "synthetic_sepsis_training.csv")
    target = "sepsis_label"
    df = df.drop(columns=["timestamp"], errors="ignore")
    num_features = [c for c in df.columns if c not in [target, "patient_id"]]
    cat_features: list[str] = []
    X_train, X_test, y_train, y_test = _split(df, target)
    X_train = X_train.drop(columns=["patient_id"], errors="ignore")
    X_test = X_test.drop(columns=["patient_id"], errors="ignore")

    pipeline = _build_pipeline(num_features, cat_features)
    pipeline.fit(X_train, y_train)
    metrics = _evaluate_binary(pipeline, X_test, y_test, "Sepsis")
    out = {
        "pipeline": pipeline,
        "num_features": num_features,
        "cat_features": cat_features,
        "target": target,
        "metrics": metrics,
        "threshold": 0.5,
    }
    joblib.dump(out, _MODELS_DIR / "sepsis_model.joblib")
    logger.info(f"Saved: {_MODELS_DIR / 'sepsis_model.joblib'}")
    return out


def train_aki():
    logger.info("Training: AKI model")
    df = pd.read_csv(_PROJECT_ROOT / "synthetic_aki_training.csv")
    target = "aki_label"
    df2 = df.drop(columns=["timestamp"], errors="ignore")
    X_train, X_test, y_train, y_test = _split(df2, target)
    num_features = [c for c in X_train.drop(columns=["patient_id"], errors="ignore").columns]
    cat_features: list[str] = []
    X_train = X_train.drop(columns=["patient_id"], errors="ignore")
    X_test = X_test.drop(columns=["patient_id"], errors="ignore")
    pipeline = _build_pipeline(num_features, cat_features)
    pipeline.fit(X_train, y_train)
    metrics = _evaluate_binary(pipeline, X_test, y_test, "AKI (Acute Kidney Injury)")
    out = {"pipeline": pipeline, "num_features": num_features, "cat_features": cat_features,
           "target": target, "metrics": metrics, "threshold": 0.5}
    joblib.dump(out, _MODELS_DIR / "aki_model.joblib")
    logger.info(f"Saved: {_MODELS_DIR / 'aki_model.joblib'}")
    return out


def train_cardio():
    logger.info("Training: Cardiovascular Risk model")
    df = pd.read_csv(_PROJECT_ROOT / "synthetic_cardiovascular_risk_training.csv")
    target = "cardiovascular_risk_label"
    df2 = df.drop(columns=["timestamp"], errors="ignore")
    X_train, X_test, y_train, y_test = _split(df2, target)
    num_features = [c for c in X_train.drop(columns=["patient_id"], errors="ignore").columns]
    cat_features: list[str] = []
    X_train = X_train.drop(columns=["patient_id"], errors="ignore")
    X_test = X_test.drop(columns=["patient_id"], errors="ignore")
    pipeline = _build_pipeline(num_features, cat_features)
    pipeline.fit(X_train, y_train)
    metrics = _evaluate_binary(pipeline, X_test, y_test, "Cardiovascular Risk")
    out = {"pipeline": pipeline, "num_features": num_features, "cat_features": cat_features,
           "target": target, "metrics": metrics, "threshold": 0.5}
    joblib.dump(out, _MODELS_DIR / "cardio_model.joblib")
    logger.info(f"Saved: {_MODELS_DIR / 'cardio_model.joblib'}")
    return out


def train_liver():
    logger.info("Training: Liver Risk model")
    df = pd.read_csv(_PROJECT_ROOT / "synthetic_liver_risk_training.csv")
    target = "liver_risk_label"
    df2 = df.drop(columns=["timestamp"], errors="ignore")
    X_train, X_test, y_train, y_test = _split(df2, target)
    num_features = [c for c in X_train.drop(columns=["patient_id"], errors="ignore").columns]
    cat_features: list[str] = []
    X_train = X_train.drop(columns=["patient_id"], errors="ignore")
    X_test = X_test.drop(columns=["patient_id"], errors="ignore")
    pipeline = _build_pipeline(num_features, cat_features)
    pipeline.fit(X_train, y_train)
    metrics = _evaluate_binary(pipeline, X_test, y_test, "Liver Risk")
    out = {"pipeline": pipeline, "num_features": num_features, "cat_features": cat_features,
           "target": target, "metrics": metrics, "threshold": 0.5}
    joblib.dump(out, _MODELS_DIR / "liver_model.joblib")
    logger.info(f"Saved: {_MODELS_DIR / 'liver_model.joblib'}")
    return out


def train_neuro():
    logger.info("Training: Neurological Risk model")
    df = pd.read_csv(_PROJECT_ROOT / "synthetic_neurological_risk_training.csv")
    target = "neurological_risk_label"
    df2 = df.drop(columns=["timestamp"], errors="ignore")
    X_train, X_test, y_train, y_test = _split(df2, target)
    num_features = [c for c in X_train.drop(columns=["patient_id"], errors="ignore").columns
                    if df2[c].dtype in [np.float64, np.int64, float, int]]
    cat_features = [c for c in X_train.drop(columns=["patient_id"], errors="ignore").columns
                    if df2[c].dtype == object]
    X_train = X_train.drop(columns=["patient_id"], errors="ignore")
    X_test = X_test.drop(columns=["patient_id"], errors="ignore")
    pipeline = _build_pipeline(num_features, cat_features)
    pipeline.fit(X_train, y_train)
    metrics = _evaluate_binary(pipeline, X_test, y_test, "Neurological Risk")
    out = {"pipeline": pipeline, "num_features": num_features, "cat_features": cat_features,
           "target": target, "metrics": metrics, "threshold": 0.5}
    joblib.dump(out, _MODELS_DIR / "neuro_model.joblib")
    logger.info(f"Saved: {_MODELS_DIR / 'neuro_model.joblib'}")
    return out


def train_resp():
    logger.info("Training: Respiratory Risk model")
    df = pd.read_csv(_PROJECT_ROOT / "synthetic_respiratory_risk_training.csv")
    target = "respiratory_risk_label"
    df2 = df.drop(columns=["timestamp"], errors="ignore")
    X_train, X_test, y_train, y_test = _split(df2, target)
    num_features = [c for c in X_train.drop(columns=["patient_id"], errors="ignore").columns
                    if df2[c].dtype in [np.float64, np.int64, float, int]]
    cat_features = [c for c in X_train.drop(columns=["patient_id"], errors="ignore").columns
                    if df2[c].dtype == object]
    X_train = X_train.drop(columns=["patient_id"], errors="ignore")
    X_test = X_test.drop(columns=["patient_id"], errors="ignore")
    pipeline = _build_pipeline(num_features, cat_features)
    pipeline.fit(X_train, y_train)
    metrics = _evaluate_binary(pipeline, X_test, y_test, "Respiratory Risk")
    out = {"pipeline": pipeline, "num_features": num_features, "cat_features": cat_features,
           "target": target, "metrics": metrics, "threshold": 0.5}
    joblib.dump(out, _MODELS_DIR / "resp_model.joblib")
    logger.info(f"Saved: {_MODELS_DIR / 'resp_model.joblib'}")
    return out


def train_antibiotic():
    logger.info("Training: Antibiotic Recommendation Demo model (SYNTHETIC — research only)")
    df = pd.read_csv(_PROJECT_ROOT / "synthetic_antibiotic.csv")
    target = "recommended_antibiotic_demo_label"

    # Exclude leakage columns
    exclude = ["patient_id", "hour", "data_origin", "label_origin", target]
    feature_df = df.drop(columns=[c for c in exclude if c in df.columns])

    num_features = [c for c in feature_df.columns if feature_df[c].dtype in [np.float64, np.int64, float, int]]
    cat_features = [c for c in feature_df.columns if feature_df[c].dtype == object]

    logger.info(f"Numerical features: {num_features}")
    logger.info(f"Categorical features: {cat_features}")

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(df[target])

    # Group split by patient_id to avoid patient overlap
    from sklearn.model_selection import GroupShuffleSplit
    groups = df["patient_id"]
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(feature_df, y_encoded, groups))

    X_train = feature_df.iloc[train_idx]
    X_test = feature_df.iloc[test_idx]
    y_train = y_encoded[train_idx]
    y_test = y_encoded[test_idx]

    pipeline = _build_pipeline(num_features, cat_features)
    pipeline.fit(X_train, y_train)
    metrics = _evaluate_multiclass(pipeline, X_test, y_test, le, "Antibiotic Demo")

    out = {
        "pipeline": pipeline,
        "label_encoder": le,
        "num_features": num_features,
        "cat_features": cat_features,
        "target": target,
        "classes": list(le.classes_),
        "metrics": metrics,
        "disclaimer": "SYNTHETIC DEMO — rule-based labels, NOT clinical ground truth",
    }
    joblib.dump(out, _MODELS_DIR / "antibiotic_model.joblib")
    logger.info(f"Saved: {_MODELS_DIR / 'antibiotic_model.joblib'}")
    return out


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = {}
    for name, fn in [
        ("sepsis", train_sepsis),
        ("aki", train_aki),
        ("cardio", train_cardio),
        ("liver", train_liver),
        ("neuro", train_neuro),
        ("resp", train_resp),
        ("antibiotic", train_antibiotic),
    ]:
        try:
            results[name] = fn()
            logger.info(f"✅ {name} model trained successfully.")
        except Exception as exc:
            logger.error(f"❌ {name} model training failed: {exc}", exc_info=True)

    logger.info("\n\n=== TRAINING SUMMARY ===")
    for name, r in results.items():
        if "metrics" in r:
            m = r["metrics"]
            roc = m.get("roc_auc", "N/A (multiclass)")
            logger.info(f"  {name}: ROC-AUC={roc}")
