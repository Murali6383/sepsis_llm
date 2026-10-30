"""
ML Model Loader and Inference Service
--------------------------------------
Loads trained joblib pipelines once at startup and exposes prediction methods.
SHAP explanations are computed using TreeExplainer when supported.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
_MODELS_DIR = _BACKEND_DIR / "models_store"

# Risk level thresholds
_RISK_THRESHOLDS = {"low": 0.35, "moderate": 0.65}


def _risk_level(prob: float) -> str:
    if prob < _RISK_THRESHOLDS["low"]:
        return "low"
    if prob < _RISK_THRESHOLDS["moderate"]:
        return "moderate"
    return "high"


def _load(filename: str) -> Optional[dict]:
    path = _MODELS_DIR / filename
    if not path.exists():
        logger.warning(f"Model file not found: {path}")
        return None
    try:
        return joblib.load(path)
    except Exception as exc:
        logger.error(f"Failed to load {path}: {exc}")
        return None


# ── SHAP ──────────────────────────────────────────────────────────────────────

def _shap_values_for_pipeline(
    artifact: dict, X: pd.DataFrame
) -> Optional[List[Dict[str, Any]]]:
    """Compute per-feature SHAP values for a single sample.
    Returns a list of {feature, shap_value, value} dicts or None on failure.
    """
    try:
        import shap

        pipeline = artifact["pipeline"]
        preprocessor = pipeline.named_steps["preprocessor"]
        clf = pipeline.named_steps["classifier"]

        X_transformed = preprocessor.transform(X)

        # Derive transformed feature names
        feature_names: List[str] = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "num":
                feature_names.extend(cols)
            elif name == "cat":
                try:
                    ohe = trans.named_steps["encoder"]
                    feature_names.extend(ohe.get_feature_names_out(cols).tolist())
                except Exception:
                    feature_names.extend([f"{name}_{i}" for i in range(X_transformed.shape[1] - len(feature_names))])

        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_transformed)

        # Binary: sv is (n_samples, n_features); Multiclass: sv is list
        if isinstance(sv, list):
            # Use positive class (index of highest predicted prob)
            proba = clf.predict_proba(X_transformed)
            cls_idx = int(np.argmax(proba[0]))
            sv_row = sv[cls_idx][0] if sv[cls_idx].ndim == 2 else sv[cls_idx]
        else:
            sv_row = sv[0] if sv.ndim == 2 else sv

        result = []
        for i, (feat, val) in enumerate(zip(feature_names, sv_row)):
            original_val = X.iloc[0].get(feat, None) if feat in X.columns else None
            result.append({
                "feature": feat,
                "shap_value": round(float(val), 6),
                "value": float(original_val) if original_val is not None else None,
            })
        # Sort by absolute SHAP value descending
        result.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return result[:15]  # Return top-15 features

    except Exception as exc:
        logger.warning(f"SHAP computation failed: {exc}")
        return None


# ── Prediction helpers ─────────────────────────────────────────────────────────

def _binary_predict(
    artifact: Optional[dict],
    features: Dict[str, Any],
    organ_name: str,
    include_shap: bool = True,
) -> Optional[Dict[str, Any]]:
    if artifact is None:
        return None
    try:
        pipeline = artifact["pipeline"]
        num_features = artifact["num_features"]
        cat_features = artifact["cat_features"]
        all_features = num_features + cat_features
        threshold = artifact.get("threshold", 0.5)

        row = {f: features.get(f, None) for f in all_features}
        X = pd.DataFrame([row])

        prob = float(pipeline.predict_proba(X)[0, 1])
        label = int(prob >= threshold)

        shap_vals = None
        if include_shap:
            shap_vals = _shap_values_for_pipeline(artifact, X)

        return {
            "organ": organ_name,
            "probability": round(prob, 4),
            "label": label,
            "threshold": threshold,
            "risk_level": _risk_level(prob),
            "model_type": "LightGBM",
            "shap_available": shap_vals is not None,
            "shap_values": shap_vals,
        }
    except Exception as exc:
        logger.error(f"{organ_name} prediction error: {exc}", exc_info=True)
        return None


# ── Public Inference API ───────────────────────────────────────────────────────

class MLInferenceService:
    """Singleton service — loads models once, exposes predict_* methods."""

    def __init__(self):
        logger.info("Loading ML model artifacts …")
        self._sepsis = _load("sepsis_model.joblib")
        self._aki = _load("aki_model.joblib")
        self._cardio = _load("cardio_model.joblib")
        self._liver = _load("liver_model.joblib")
        self._neuro = _load("neuro_model.joblib")
        self._resp = _load("resp_model.joblib")
        self._antibiotic = _load("antibiotic_model.joblib")
        self._log_loaded_models()

    def _log_loaded_models(self):
        loaded = [
            n for n, a in [
                ("sepsis", self._sepsis), ("aki", self._aki), ("cardio", self._cardio),
                ("liver", self._liver), ("neuro", self._neuro), ("resp", self._resp),
                ("antibiotic", self._antibiotic),
            ] if a is not None
        ]
        logger.info(f"Models loaded: {loaded}")

    # ── Sepsis ─────────────────────────────────────────────────────────────

    def predict_sepsis(self, features: Dict[str, Any], include_shap: bool = True) -> Optional[Dict[str, Any]]:
        result = _binary_predict(self._sepsis, features, "sepsis", include_shap)
        if result:
            result["model_type"] = "LightGBM"
            result["sepsis_probability"] = result.pop("probability")
            result["sepsis_label"] = result.pop("label")
        return result

    # ── Organ Models ───────────────────────────────────────────────────────

    def predict_aki(self, features: Dict[str, Any], include_shap: bool = True) -> Optional[Dict[str, Any]]:
        return _binary_predict(self._aki, features, "acute_kidney_injury", include_shap)

    def predict_cardio(self, features: Dict[str, Any], include_shap: bool = True) -> Optional[Dict[str, Any]]:
        return _binary_predict(self._cardio, features, "cardiovascular", include_shap)

    def predict_liver(self, features: Dict[str, Any], include_shap: bool = True) -> Optional[Dict[str, Any]]:
        return _binary_predict(self._liver, features, "liver", include_shap)

    def predict_neuro(self, features: Dict[str, Any], include_shap: bool = True) -> Optional[Dict[str, Any]]:
        return _binary_predict(self._neuro, features, "neurological", include_shap)

    def predict_resp(self, features: Dict[str, Any], include_shap: bool = True) -> Optional[Dict[str, Any]]:
        return _binary_predict(self._resp, features, "respiratory", include_shap)

    # ── Antibiotic ─────────────────────────────────────────────────────────

    def predict_antibiotic(self, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        artifact = self._antibiotic
        if artifact is None:
            return None
        try:
            pipeline = artifact["pipeline"]
            le = artifact["label_encoder"]
            num_features = artifact["num_features"]
            cat_features = artifact["cat_features"]
            all_features = num_features + cat_features
            classes = artifact["classes"]

            row = {f: features.get(f, None) for f in all_features}
            X = pd.DataFrame([row])

            proba = pipeline.predict_proba(X)[0]
            pred_idx = int(np.argmax(proba))
            pred_class = le.inverse_transform([pred_idx])[0]

            prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
            return {
                "predicted_class": pred_class,
                "probabilities": prob_dict,
                "model_type": "LightGBM (multiclass)",
                "data_disclaimer": (
                    "⚠️ RESEARCH DEMONSTRATION ONLY. Labels are rule-based synthetic data "
                    "(label_origin: rule_table_demo_not_clinical_ground_truth). "
                    "This model must NOT be used for clinical decisions."
                ),
            }
        except Exception as exc:
            logger.error(f"Antibiotic prediction error: {exc}", exc_info=True)
            return None

    def models_available(self) -> Dict[str, bool]:
        return {
            "sepsis": self._sepsis is not None,
            "aki": self._aki is not None,
            "cardiovascular": self._cardio is not None,
            "liver": self._liver is not None,
            "neurological": self._neuro is not None,
            "respiratory": self._resp is not None,
            "antibiotic": self._antibiotic is not None,
        }


@lru_cache(maxsize=1)
def get_ml_service() -> MLInferenceService:
    return MLInferenceService()
