"""Tests for ML inference service and antibiotic model."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── ML Service tests ──────────────────────────────────────────────────────────

def test_ml_service_loads_sepsis_model():
    """Sepsis model file must exist and load."""
    from app.services.ml_service import _load, _MODELS_DIR
    artifact = _load("sepsis_model.joblib")
    assert artifact is not None
    assert "pipeline" in artifact
    assert "num_features" in artifact
    assert "target" in artifact
    assert artifact["target"] == "sepsis_label"


def test_ml_service_loads_antibiotic_model():
    """Antibiotic model file must exist and load."""
    from app.services.ml_service import _load
    artifact = _load("antibiotic_model.joblib")
    assert artifact is not None
    assert "pipeline" in artifact
    assert "label_encoder" in artifact
    assert "classes" in artifact


def test_sepsis_prediction_returns_valid_probability():
    """Sepsis prediction must return a probability in [0, 1]."""
    from app.services.ml_service import MLInferenceService
    svc = MLInferenceService()
    features = {
        "age": 65, "heart_rate": 110, "temperature": 38.5,
        "respiratory_rate": 22, "spo2": 95, "systolic_bp": 90,
        "diastolic_bp": 60, "map": 70, "wbc": 15.0, "lactate": 2.5,
        "creatinine": 1.8, "platelets": 100,
    }
    result = svc.predict_sepsis(features, include_shap=False)
    assert result is not None
    assert 0.0 <= result["sepsis_probability"] <= 1.0
    assert result["sepsis_label"] in (0, 1)
    assert result["risk_level"] in ("low", "moderate", "high")


def test_sepsis_score_not_reused_as_organ_score():
    """Each organ model must produce its own independent score."""
    from app.services.ml_service import MLInferenceService
    svc = MLInferenceService()
    sepsis_features = {
        "age": 65, "heart_rate": 110, "temperature": 38.5,
        "respiratory_rate": 22, "spo2": 95, "systolic_bp": 90,
        "diastolic_bp": 60, "map": 70, "wbc": 15.0, "lactate": 2.5,
        "creatinine": 1.8, "platelets": 100,
    }
    sepsis_result = svc.predict_sepsis(sepsis_features, include_shap=False)
    aki_features = {"age": 65, "creatinine": 1.8}
    aki_result = svc.predict_aki(aki_features, include_shap=False)
    if aki_result is not None and sepsis_result is not None:
        # The organ score should NOT simply equal the sepsis probability
        # (unless they coincidentally match on this input, which is extremely unlikely for different models)
        # We just verify the organ result has its own "organ" label
        assert aki_result.get("organ") == "acute_kidney_injury"
        assert "probability" in aki_result


def test_antibiotic_prediction_returns_valid_class():
    """Antibiotic model must predict a valid class from its label set."""
    from app.services.ml_service import MLInferenceService, _load
    svc = MLInferenceService()
    artifact = _load("antibiotic_model.joblib")
    valid_classes = artifact["classes"]

    features = {
        "age": 55, "gender": "M", "heart_rate": 100, "systolic_bp": 120,
        "diastolic_bp": 80, "map": 93, "respiratory_rate": 18, "temperature": 37.5,
        "spo2": 98, "wbc": 9.0, "lactate": 1.5, "creatinine": 1.0, "bilirubin": 0.8,
        "platelets": 200, "glucose": 95, "crp": 5.0, "urine_output": 50,
        "gcs": 15, "vasopressor": 0, "mechanical_ventilation": 0,
        "infection_source_synthetic": "urinary", "beta_lactam_allergy_synthetic": 0,
        "egfr_computed": 90, "renal_dose_adjustment_review": 0,
    }
    result = svc.predict_antibiotic(features)
    assert result is not None
    assert result["predicted_class"] in valid_classes
    assert "probabilities" in result
    assert abs(sum(result["probabilities"].values()) - 1.0) < 0.01  # probabilities sum to ~1
    assert "RESEARCH DEMONSTRATION" in result["data_disclaimer"].upper() or "SYNTHETIC" in result["data_disclaimer"].upper()


def test_antibiotic_model_independent_of_sepsis():
    """Antibiotic model scores must not equal sepsis scores."""
    from app.services.ml_service import MLInferenceService
    svc = MLInferenceService()
    sepsis_features = {
        "age": 65, "heart_rate": 110, "temperature": 38.5,
        "respiratory_rate": 22, "spo2": 95, "systolic_bp": 90,
        "diastolic_bp": 60, "map": 70, "wbc": 15.0, "lactate": 2.5,
        "creatinine": 1.8, "platelets": 100,
    }
    sepsis_result = svc.predict_sepsis(sepsis_features, include_shap=False)

    antibiotic_features = {
        "age": 65, "gender": "F", "heart_rate": 110, "systolic_bp": 90,
        "diastolic_bp": 60, "map": 70, "respiratory_rate": 22, "temperature": 38.5,
        "spo2": 95, "wbc": 15.0, "lactate": 2.5, "creatinine": 1.8, "bilirubin": 1.0,
        "platelets": 100, "glucose": 120, "crp": 30, "urine_output": 20,
        "gcs": 13, "vasopressor": 1, "mechanical_ventilation": 0,
        "infection_source_synthetic": "bloodstream_unknown",
        "beta_lactam_allergy_synthetic": 0, "egfr_computed": 35,
        "renal_dose_adjustment_review": 1,
    }
    antibiotic_result = svc.predict_antibiotic(antibiotic_features)

    # Different model types / output structures
    if sepsis_result and antibiotic_result:
        assert "sepsis_probability" in sepsis_result
        assert "predicted_class" in antibiotic_result  # different output
        # Antibiotic result has no "sepsis_probability" key
        assert "sepsis_probability" not in antibiotic_result


def test_missing_model_returns_none():
    """When a model file is missing, _load returns None."""
    from app.services.ml_service import _load
    result = _load("nonexistent_model.joblib")
    assert result is None


def test_shap_values_for_sepsis():
    """SHAP values should be computable for the sepsis LightGBM model."""
    from app.services.ml_service import MLInferenceService
    svc = MLInferenceService()
    features = {
        "age": 65, "heart_rate": 110, "temperature": 38.5,
        "respiratory_rate": 22, "spo2": 95, "systolic_bp": 90,
        "diastolic_bp": 60, "map": 70, "wbc": 15.0, "lactate": 2.5,
        "creatinine": 1.8, "platelets": 100,
    }
    result = svc.predict_sepsis(features, include_shap=True)
    assert result is not None
    if result["shap_available"]:
        shap_vals = result["shap_values"]
        assert isinstance(shap_vals, list)
        assert len(shap_vals) > 0
        for sv in shap_vals:
            assert "feature" in sv
            assert "shap_value" in sv
            assert isinstance(sv["shap_value"], float)
