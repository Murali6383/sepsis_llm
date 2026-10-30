"""AI/LLM routes — explanation, chat, report, health, antibiotic."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.core.config import get_settings
from app.core.security import TokenPayload, get_current_user
from app.schemas.schemas import (
    AntibioticFeaturesRequest,
    AntibioticPredictionResponse,
    AIHealthResponse,
    ChatRequest,
    ChatResponse,
    ExplainRequest,
    ExplainResponse,
    ReportRequest,
    ReportResponse,
)
from app.services.groq_service import get_groq_service
from app.services.ml_service import get_ml_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ai", tags=["AI / LLM"])

# Request body size limit (bytes)
_MAX_BODY_BYTES = 32_768  # 32 KB


def _check_body_size(request: Request):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > _MAX_BODY_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Request body too large.",
        )


# ── Health ─────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=AIHealthResponse)
def ai_health():
    """
    Report LLM configuration status.
    Does NOT verify live provider connectivity — reports configuration only.
    """
    settings = get_settings()
    groq = get_groq_service()
    return AIHealthResponse(
        llm_configured=groq.is_configured(),
        llm_model=settings.GROQ_MODEL,
        groq_api_key_present=settings.groq_configured,
        note=(
            "LLM is ready." if groq.is_configured()
            else "Set GROQ_API_KEY in backend/.env to enable LLM features. ML predictions remain functional."
        ),
    )


# ── Explain ─────────────────────────────────────────────────────────────────────

@router.post("/explain", response_model=ExplainResponse)
def explain_prediction(
    request: ExplainRequest,
    user: TokenPayload = Depends(get_current_user),
):
    """
    Generate an LLM explanation from trusted ML model results.
    The LLM explains model outputs only — it does NOT calculate new scores.
    """
    groq = get_groq_service()
    if not groq.is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Groq LLM not configured. Set GROQ_API_KEY in backend/.env.",
        )
    context = {
        "sepsis_result": request.sepsis_result,
        "organ_results": request.organ_results or [],
        "antibiotic_result": request.antibiotic_result,
        "shap_values": request.shap_values or [],
        "missing_features": request.missing_features or [],
    }
    try:
        explanation = groq.explain_prediction(context)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    return ExplainResponse(explanation=explanation)


# ── Chat ─────────────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    user: TokenPayload = Depends(get_current_user),
):
    """
    Patient-context-aware AI chat assistant.
    Context is supplied by the client; predictions are NOT re-run here.
    The LLM explains existing outputs only.
    """
    if not request.message.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Message cannot be empty.",
        )

    groq = get_groq_service()
    if not groq.is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Groq LLM not configured. Set GROQ_API_KEY in backend/.env.",
        )

    # Build context from patient_context dict (de-identified)
    from app.services.groq_service import _build_context_text
    ctx_text = _build_context_text(request.patient_context) if request.patient_context else None

    history = [{"role": m.role, "content": m.content} for m in request.history]

    try:
        answer = groq.chat(
            user_message=request.message,
            history=history,
            extra_context=ctx_text,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )

    return ChatResponse(answer=answer)


# ── Report ─────────────────────────────────────────────────────────────────────

@router.post("/report", response_model=ReportResponse)
def generate_report(
    request: ReportRequest,
    user: TokenPayload = Depends(get_current_user),
):
    """
    Generate a structured AI research report from actual model outputs.
    Does not generate fake values, citations, diagnoses, or treatment recommendations.
    """
    groq = get_groq_service()
    if not groq.is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Groq LLM not configured. Set GROQ_API_KEY in backend/.env.",
        )

    context = {
        "sepsis_result": request.sepsis_result,
        "organ_results": request.organ_results or [],
        "antibiotic_result": request.antibiotic_result,
        "shap_values": request.shap_values or [],
    }

    try:
        report_text = groq.generate_report(context, patient_ref=request.patient_ref)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )

    return ReportResponse(
        report_text=report_text,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ── Antibiotic Suitability ────────────────────────────────────────────────────────

@router.post("/antibiotic-suitability", response_model=AntibioticPredictionResponse)
def antibiotic_suitability(
    request: AntibioticFeaturesRequest,
    user: TokenPayload = Depends(get_current_user),
):
    """
    Predict antibiotic recommendation using a LightGBM multiclass model.

    ⚠️ RESEARCH DEMONSTRATION ONLY.
    The training labels are rule-based synthetic data
    (label_origin: rule_table_demo_not_clinical_ground_truth).
    This endpoint must NOT be used for clinical decisions.
    """
    ml = get_ml_service()
    features = request.model_dump()
    result = ml.predict_antibiotic(features)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Antibiotic model is not available. "
                "Run training first: python -m app.models.train_models"
            ),
        )
    return AntibioticPredictionResponse(**result)
