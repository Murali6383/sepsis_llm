"""Tests for Groq LLM service."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.groq_service import GroqService, _build_context_text


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_service(api_key: str = "", model: str = "llama3-8b-8192") -> GroqService:
    """Create a GroqService with mocked settings."""
    with patch("app.services.groq_service.get_settings") as mock_settings:
        settings = MagicMock()
        settings.GROQ_API_KEY = api_key
        settings.GROQ_MODEL = model
        settings.GROQ_TIMEOUT_SECONDS = 10
        settings.GROQ_MAX_TOKENS = 500
        settings.groq_configured = bool(api_key and api_key != "your_groq_api_key_here")
        mock_settings.return_value = settings
        svc = GroqService.__new__(GroqService)
        svc._settings = settings
        svc._client = None
        return svc


# ── Configuration tests ────────────────────────────────────────────────────────

def test_groq_service_not_configured_when_key_missing():
    svc = _make_service(api_key="")
    assert not svc.is_configured()


def test_groq_service_not_configured_when_key_is_placeholder():
    svc = _make_service(api_key="your_groq_api_key_here")
    assert not svc.is_configured()


def test_groq_service_raises_when_not_configured():
    svc = _make_service(api_key="")
    with pytest.raises(RuntimeError, match="not configured"):
        svc.chat("Hello")


# ── Mocked response tests ──────────────────────────────────────────────────────

def test_groq_chat_returns_text_on_success():
    svc = _make_service(api_key="test-key-xyz")
    svc._settings.groq_configured = True

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "This is a test explanation."
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    svc._client = mock_client

    result = svc.chat("Explain the sepsis result.")
    assert "test explanation" in result
    assert mock_client.chat.completions.create.called


def test_groq_chat_raises_on_permanent_401():
    svc = _make_service(api_key="bad-key")
    svc._settings.groq_configured = True

    class AuthError(Exception):
        status_code = 401

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = AuthError("Unauthorized")
    svc._client = mock_client

    with pytest.raises(RuntimeError, match="authentication"):
        svc.chat("Test")


def test_groq_chat_retries_on_transient_error_then_succeeds():
    svc = _make_service(api_key="test-key")
    svc._settings.groq_configured = True

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Success after retry."

    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            e = Exception("Transient network error")
            e.status_code = 500
            raise e
        return mock_response

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = side_effect
    svc._client = mock_client

    with patch("app.services.groq_service.time.sleep"):  # skip sleep in test
        result = svc.chat("Test")
    assert "Success after retry" in result


def test_groq_chat_raises_after_max_retries():
    svc = _make_service(api_key="test-key")
    svc._settings.groq_configured = True

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("Network failure")
    svc._client = mock_client

    with patch("app.services.groq_service.time.sleep"):
        with pytest.raises(RuntimeError, match="unavailable"):
            svc.chat("Test")


def test_groq_explain_prediction_builds_context():
    svc = _make_service(api_key="test-key")
    svc._settings.groq_configured = True

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Explanation text."
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    svc._client = mock_client

    context = {
        "sepsis_result": {"sepsis_probability": 0.82, "sepsis_label": 1, "risk_level": "high"},
        "organ_results": [],
        "antibiotic_result": None,
        "shap_values": [],
    }
    result = svc.explain_prediction(context)
    assert isinstance(result, str)
    assert len(result) > 0


# ── Context building tests ─────────────────────────────────────────────────────

def test_build_context_text_includes_sepsis():
    ctx = {
        "sepsis_result": {"sepsis_probability": 0.75, "sepsis_label": 1, "risk_level": "high"},
        "organ_results": [],
        "antibiotic_result": None,
        "shap_values": [],
        "missing_features": ["lactate"],
    }
    text = _build_context_text(ctx)
    assert "0.75" in text
    assert "SEPSIS" in text
    assert "lactate" in text


def test_build_context_text_includes_antibiotic_disclaimer():
    ctx = {
        "sepsis_result": None,
        "organ_results": [],
        "antibiotic_result": {
            "predicted_class": "ceftriaxone",
            "data_disclaimer": "SYNTHETIC",
        },
        "shap_values": [],
    }
    text = _build_context_text(ctx)
    assert "ceftriaxone" in text
    assert "SYNTHETIC" in text.upper() or "synthetic" in text.lower()


def test_sepsis_score_not_reused_as_antibiotic_score():
    """Ensure sepsis probability and antibiotic prediction are separate in context."""
    ctx = {
        "sepsis_result": {"sepsis_probability": 0.9, "sepsis_label": 1},
        "antibiotic_result": {"predicted_class": "meropenem+vancomycin"},
        "organ_results": [],
        "shap_values": [],
    }
    text = _build_context_text(ctx)
    # The antibiotic section should NOT reference the sepsis probability value directly
    antibiotic_section_start = text.find("ANTIBIOTIC")
    if antibiotic_section_start >= 0:
        antibiotic_section = text[antibiotic_section_start:]
        # 0.9 should NOT appear in the antibiotic section
        assert "0.9" not in antibiotic_section, \
            "Sepsis probability must not appear in antibiotic section."
