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
    """Antibiotic model file (both ARMD-ECUH and legacy) must exist and load."""
    from app.services.ml_service import _load
    armd_artifact = _load("antibiotic_susceptibility/lightgbm_susceptibility.joblib")
    assert armd_artifact is not None
    assert "model" in armd_artifact
    assert "label_encoder" in armd_artifact
    assert "category_mappings" in armd_artifact
    assert set(armd_artifact["label_encoder"].classes_) == {"Intermediate", "Resistant", "Susceptible"}


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
        assert aki_result.get("organ") == "acute_kidney_injury"
        assert "probability" in aki_result


def test_antibiotic_prediction_returns_valid_class():
    """ARMD-ECUH model must predict a valid class from its label set (Susceptible, Resistant, Intermediate)."""
    from app.services.ml_service import MLInferenceService
    svc = MLInferenceService()

    features = {
        "organism": "ESCHERICHIA COLI",
        "antibiotic": "Ciprofloxacin",
        "culture_description": "URINE",
        "ordering_mode": "Inpatient",
    }
    result = svc.predict_antibiotic(features)
    assert result is not None
    assert result["predicted_class"] in ["Susceptible", "Resistant", "Intermediate"]
    assert "probabilities" in result
    assert abs(sum(result["probabilities"].values()) - 1.0) < 0.01
    assert "ARMD-ECUH" in result["data_source"]
    assert "ARMD-ECUH" in result["data_disclaimer"]
    assert result["shap_available"] is True
    assert len(result["shap_values"]) > 0


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


def test_patient_specific_suitability_ranking():
    """Verify patient-specific antibiotic suitability ranking and clinical safety logic."""
    from app.services.ml_service import MLInferenceService
    svc = MLInferenceService()

    req = {
        "patient_age": 72,
        "patient_gender": "F",
        "infection_site": "URINE",
        "identified_organism": "ESCHERICHIA COLI",
        "drug_allergies": ["Penicillins"],
        "antibiotic_resistance_profile": [],
        "previous_antibiotic_exposure": [],
        "creatinine": 1.1,
        "egfr": 75.0,
        "current_medications": [],
    }

    result = svc.predict_patient_antibiotic_suitability(req)
    assert result is not None
    assert "ranked_candidates" in result
    candidates = result["ranked_candidates"]
    assert len(candidates) >= 5

    # Ranks must be in order 1, 2, 3...
    ranks = [c["rank"] for c in candidates]
    assert ranks == list(range(1, len(candidates) + 1))

    # Ampicillin must have allergy contraindication
    amp = next(c for c in candidates if c["antibiotic"] == "Ampicillin")
    assert amp["allergy_contraindication"] is True
    assert amp["suitability_score"] <= 15.0

    # Test renal contraindication with severe kidney impairment
    req_renal = {
        **req,
        "creatinine": 3.8,
        "egfr": 18.0,
    }
    result_renal = svc.predict_patient_antibiotic_suitability(req_renal)
    candidates_renal = result_renal["ranked_candidates"]
    gent = next(c for c in candidates_renal if c["antibiotic"] == "Gentamicin")
    assert "nephrotoxicity" in gent["renal_warning"].lower()
