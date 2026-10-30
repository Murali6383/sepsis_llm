"""API integration tests using FastAPI TestClient."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app
from app.core.security import create_access_token

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def doctor_token():
    return create_access_token("test_doctor", role="doctor")


@pytest.fixture
def auth_headers(doctor_token):
    return {"Authorization": f"Bearer {doctor_token}"}


# ── Auth tests ─────────────────────────────────────────────────────────────────

def test_login_success(client):
    resp = client.post("/api/auth/login", json={"username": "doctor", "password": "doctor123"})
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["role"] == "doctor"


def test_login_invalid_credentials(client):
    resp = client.post("/api/auth/login", json={"username": "doctor", "password": "wrong"})
    assert resp.status_code == 401


def test_login_returns_no_secret(client):
    resp = client.post("/api/auth/login", json={"username": "doctor", "password": "doctor123"})
    body = resp.text
    assert "GROQ_API_KEY" not in body
    assert "groq_api_key" not in body.lower()


def test_unauthenticated_request_returns_401(client):
    resp = client.post("/api/predict/sepsis", json={"age": 50})
    assert resp.status_code == 401


# ── Health ─────────────────────────────────────────────────────────────────────

def test_system_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ai_health_no_key(client):
    """AI health should report not-configured when key is absent."""
    with patch("app.api.routes.ai.get_settings") as mock_settings:
        settings = MagicMock()
        settings.groq_configured = False
        settings.GROQ_MODEL = "llama3-8b-8192"
        mock_settings.return_value = settings
        with patch("app.api.routes.ai.get_groq_service") as mock_groq:
            svc = MagicMock()
            svc.is_configured.return_value = False
            mock_groq.return_value = svc
            resp = client.get("/api/ai/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["llm_configured"] is False
    # Key value must not be revealed — groq_api_key_present is a bool (True/False), not the key itself
    assert isinstance(data.get("groq_api_key_present"), bool)
    # The actual key string must never appear in the response body
    assert "sk-" not in resp.text


# ── Prediction tests ───────────────────────────────────────────────────────────

def test_sepsis_prediction_valid(client, auth_headers):
    payload = {
        "age": 65, "heart_rate": 110, "temperature": 38.5,
        "respiratory_rate": 22, "spo2": 95, "systolic_bp": 90,
        "diastolic_bp": 60, "map": 70, "wbc": 15.0, "lactate": 2.5,
        "creatinine": 1.8, "platelets": 100,
    }
    resp = client.post("/api/predict/sepsis", json=payload, headers=auth_headers)
    assert resp.status_code in (200, 503)  # 503 if model not loaded
    if resp.status_code == 200:
        data = resp.json()
        assert 0.0 <= data["sepsis_probability"] <= 1.0
        assert data["sepsis_label"] in (0, 1)
        assert data["risk_level"] in ("low", "moderate", "high")


def test_sepsis_prediction_invalid_age(client, auth_headers):
    payload = {"age": 200}  # exceeds max
    resp = client.post("/api/predict/sepsis", json=payload, headers=auth_headers)
    assert resp.status_code == 422


def test_antibiotic_prediction_valid(client, auth_headers):
    payload = {
        "age": 55, "gender": "M", "heart_rate": 100, "systolic_bp": 120,
        "diastolic_bp": 80, "map": 93, "respiratory_rate": 18, "temperature": 37.5,
        "spo2": 98, "wbc": 9.0, "lactate": 1.5, "creatinine": 1.0, "bilirubin": 0.8,
        "platelets": 200, "glucose": 95, "crp": 5.0, "urine_output": 50,
        "gcs": 15, "vasopressor": 0, "mechanical_ventilation": 0,
        "infection_source_synthetic": "urinary", "beta_lactam_allergy_synthetic": 0,
        "egfr_computed": 90, "renal_dose_adjustment_review": 0,
    }
    resp = client.post("/api/ai/antibiotic-suitability", json=payload, headers=auth_headers)
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        data = resp.json()
        assert "predicted_class" in data
        assert "RESEARCH" in data["data_disclaimer"].upper() or "SYNTHETIC" in data["data_disclaimer"].upper()


def test_antibiotic_invalid_infection_source(client, auth_headers):
    payload = {
        "age": 55, "gender": "M", "gcs": 15, "vasopressor": 0,
        "mechanical_ventilation": 0, "infection_source_synthetic": "invalid_source",
        "beta_lactam_allergy_synthetic": 0, "renal_dose_adjustment_review": 0,
    }
    resp = client.post("/api/ai/antibiotic-suitability", json=payload, headers=auth_headers)
    assert resp.status_code == 422


# ── AI / LLM route tests ────────────────────────────────────────────────────────

def test_explain_requires_auth(client):
    resp = client.post("/api/ai/explain", json={})
    assert resp.status_code == 401


def test_explain_returns_503_when_groq_unconfigured(client, auth_headers):
    with patch("app.api.routes.ai.get_groq_service") as mock_groq:
        svc = MagicMock()
        svc.is_configured.return_value = False
        mock_groq.return_value = svc
        resp = client.post("/api/ai/explain", json={}, headers=auth_headers)
    assert resp.status_code == 503


def test_explain_returns_explanation_with_mocked_groq(client, auth_headers):
    with patch("app.api.routes.ai.get_groq_service") as mock_groq:
        svc = MagicMock()
        svc.is_configured.return_value = True
        svc.explain_prediction.return_value = "Mocked LLM explanation."
        mock_groq.return_value = svc
        payload = {
            "sepsis_result": {"sepsis_probability": 0.82, "sepsis_label": 1, "risk_level": "high"},
        }
        resp = client.post("/api/ai/explain", json=payload, headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "explanation" in data
    assert len(data["explanation"]) > 0
    # Disclaimer must be present
    assert "disclaimer" in data


def test_chat_empty_message_returns_422(client, auth_headers):
    resp = client.post(
        "/api/ai/chat",
        json={"message": "   ", "patient_context": {}},
        headers=auth_headers,
    )
    assert resp.status_code == 422


def test_chat_groq_unavailable_returns_503(client, auth_headers):
    with patch("app.api.routes.ai.get_groq_service") as mock_groq:
        svc = MagicMock()
        svc.is_configured.return_value = False
        mock_groq.return_value = svc
        resp = client.post(
            "/api/ai/chat",
            json={"message": "Explain this.", "patient_context": {}},
            headers=auth_headers,
        )
    assert resp.status_code == 503


def test_report_generation_with_mocked_groq(client, auth_headers):
    with patch("app.api.routes.ai.get_groq_service") as mock_groq:
        svc = MagicMock()
        svc.is_configured.return_value = True
        svc.generate_report.return_value = "Mocked research report content."
        mock_groq.return_value = svc
        payload = {
            "sepsis_result": {"sepsis_probability": 0.75, "sepsis_label": 1},
            "patient_ref": "PT-ANON-001",
        }
        resp = client.post("/api/ai/report", json=payload, headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "report_text" in data
    assert "generated_at" in data
    assert "RESEARCH" in data["disclaimer"].upper()


def test_groq_api_key_not_exposed_in_any_response(client, auth_headers):
    """Security test: GROQ_API_KEY must never appear in any API response."""
    with patch("app.api.routes.ai.get_groq_service") as mock_groq:
        svc = MagicMock()
        svc.is_configured.return_value = True
        svc.explain_prediction.return_value = "Safe explanation."
        mock_groq.return_value = svc
        resp = client.post("/api/ai/explain", json={}, headers=auth_headers)
        body = resp.text
        assert "GROQ_API_KEY" not in body
        assert "sk-" not in body  # common API key prefix patterns
