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
        self._armd_susceptibility = _load("antibiotic_susceptibility/lightgbm_susceptibility.joblib")
        self._log_loaded_models()

    def _log_loaded_models(self):
        loaded = [
            n for n, a in [
                ("sepsis", self._sepsis), ("aki", self._aki), ("cardio", self._cardio),
                ("liver", self._liver), ("neuro", self._neuro), ("resp", self._resp),
                ("synthetic_antibiotic", self._antibiotic),
                ("armd_susceptibility", self._armd_susceptibility),
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

    # ── Antibiotic / Antimicrobial Susceptibility ──────────────────────────

    def predict_antibiotic(self, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Predict antimicrobial susceptibility using the ARMD-ECUH LightGBM model.
        Falls back to legacy synthetic pipeline only if ARMD model is unavailable.
        """
        # 1. Prefer the real ARMD-ECUH model
        if self._armd_susceptibility is not None:
            try:
                bundle = self._armd_susceptibility
                model = bundle["model"]
                feature_columns = bundle["feature_columns"]
                category_mappings = bundle["category_mappings"]
                classes = list(bundle["label_encoder"].classes_)

                # Map input values
                org_in = str(features.get("organism") or "ESCHERICHIA COLI").strip().upper()
                drug_in = str(features.get("antibiotic") or "Ciprofloxacin").strip()
                # Normalize drug title case if needed
                drug_in_title = drug_in.title()
                if drug_in_title in category_mappings.get("antibiotic", {}):
                    drug_in = drug_in_title

                culture_in = str(features.get("culture_description") or "URINE").strip().upper()
                mode_in = str(features.get("ordering_mode") or "Inpatient").strip()

                context_dict = {
                    "organism": org_in,
                    "antibiotic": drug_in,
                    "culture_description": culture_in,
                    "ordering_mode": mode_in,
                }

                # Encode ordinal values (unknown mapped to -1)
                encoded_row = {}
                for col in feature_columns:
                    val = context_dict.get(col, "UNKNOWN")
                    encoded_row[col] = [category_mappings[col].get(val, -1)]

                X = pd.DataFrame(encoded_row)
                proba = model.predict_proba(X)[0]
                pred_idx = int(np.argmax(proba))
                pred_class = classes[pred_idx]

                prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}

                # Compute genuine SHAP attribution for ARMD-ECUH model
                shap_list = None
                try:
                    import shap
                    explainer = shap.TreeExplainer(model)
                    sv = explainer.shap_values(X)
                    if isinstance(sv, np.ndarray) and sv.ndim == 3:
                        class_sv = sv[0, :, pred_idx]
                    elif isinstance(sv, list):
                        class_sv = sv[pred_idx][0] if sv[pred_idx].ndim == 2 else sv[pred_idx]
                    else:
                        class_sv = sv[0]

                    shap_list = [
                        {
                            "feature": f,
                            "shap_value": round(float(v), 4),
                            "value": context_dict.get(f),
                        }
                        for f, v in zip(feature_columns, class_sv)
                    ]
                    # Sort by absolute SHAP magnitude descending
                    shap_list.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
                except Exception as shap_exc:
                    logger.warning(f"TreeSHAP calculation for ARMD model failed: {shap_exc}")

                return {
                    "predicted_class": pred_class,
                    "probabilities": prob_dict,
                    "model_type": "LightGBM Multiclass Classifier (ARMD-ECUH)",
                    "model_version": "v1.0-armd",
                    "data_source": "ARMD-ECUH (ECU Health, N=1,931,685 culture tests)",
                    "organism": org_in,
                    "antibiotic": drug_in,
                    "culture_description": culture_in,
                    "ordering_mode": mode_in,
                    "shap_available": shap_list is not None,
                    "shap_values": shap_list,
                    "metrics_summary": {
                        "accuracy": 0.685,
                        "balanced_accuracy": 0.702,
                        "macro_roc_auc": 0.873,
                        "test_records": 390979,
                        "test_patients": 12478,
                    },
                    "data_disclaimer": (
                        "Data source: ARMD-ECUH retrospective research dataset. "
                        "Model outputs are research estimates and have not been prospectively or clinically validated. "
                        "Susceptibility prediction is an in-vitro phenotype classification and does NOT constitute "
                        "an individualized antibiotic recommendation."
                    ),
                }
            except Exception as exc:
                logger.error(f"ARMD susceptibility prediction error: {exc}", exc_info=True)

        # 2. Fallback to legacy synthetic model if ARMD model is not loaded
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
                "model_type": "LightGBM (Synthetic Demo)",
                "model_version": "v0.9-synthetic",
                "data_source": "Synthetic Benchmark Prototype",
                "shap_available": False,
                "shap_values": None,
                "data_disclaimer": (
                    "Data source: Benchmark prototype dataset. "
                    "Model outputs are research estimates and have not been prospectively or clinically validated."
                ),
            }
        except Exception as exc:
            logger.error(f"Fallback antibiotic prediction error: {exc}", exc_info=True)
            return None

    def models_available(self) -> Dict[str, bool]:
        return {
            "sepsis": self._sepsis is not None,
            "aki": self._aki is not None,
            "cardiovascular": self._cardio is not None,
            "liver": self._liver is not None,
            "neurological": self._neuro is not None,
            "respiratory": self._resp is not None,
            "antibiotic": (self._armd_susceptibility is not None or self._antibiotic is not None),
            "armd_susceptibility": self._armd_susceptibility is not None,
        }


@lru_cache(maxsize=1)
def get_ml_service() -> MLInferenceService:
    return MLInferenceService()
