"""AI/LLM routes — explanation, chat, report, health, antibiotic."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

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
    PDFReportRequest,
    ReportRequest,
    ReportResponse,
)
from app.services.groq_service import get_groq_service
from app.services.ml_service import get_ml_service
from app.services.pdf_report_service import (
    generate_clinical_pdf_report,
    load_model_comparison,
    load_armd_metrics,
)

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


# ── PDF Report Generation ──────────────────────────────────────────────────────

@router.post("/report/pdf")
def generate_pdf_report(
    request: PDFReportRequest,
    user: TokenPayload = Depends(get_current_user),
):
    """
    Generate a professional downloadable clinical research PDF report.
    Includes actual model predictions, ARMD-ECUH susceptibility, SHAP attributions,
    and empirical multi-model benchmark evaluation tables.
    """
    try:
        pdf_bytes = generate_clinical_pdf_report(
            sepsis_result=request.sepsis_result,
            organ_results=request.organ_results,
            antibiotic_result=request.antibiotic_result,
            shap_values=request.shap_values,
            patient_ref=request.patient_ref,
            input_features=request.input_features,
        )
    except Exception as exc:
        logger.error(f"PDF generation error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate PDF report: {str(exc)}",
        )

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"sepsis_guardian_report_{timestamp_str}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )


@router.get("/report/model-evaluation")
def get_model_evaluation(user: TokenPayload = Depends(get_current_user)):
    """
    Return empirical benchmark comparison results and ARMD-ECUH test metrics.
    Compares LightGBM, XGBoost, Random Forest, and Logistic Regression.
    """
    comparison = load_model_comparison()
    metrics = load_armd_metrics()
    return {
        "model_comparison": comparison,
        "armd_metrics": metrics,
    }


# ── Antibiotic / Antimicrobial Susceptibility ──────────────────────────────────

@router.post("/antibiotic-suitability", response_model=AntibioticPredictionResponse)
@router.post("/antimicrobial-susceptibility", response_model=AntibioticPredictionResponse)
def antibiotic_suitability(
    request: AntibioticFeaturesRequest,
    user: TokenPayload = Depends(get_current_user),
):
    """
    Predict antimicrobial susceptibility (Susceptible, Resistant, Intermediate)
    using the LightGBM model trained on the ARMD-ECUH clinical microbiology cohort.

    Data source: ARMD-ECUH retrospective research dataset (ECU Health).
    Outputs are research estimates and have not been prospectively or clinically validated.
    """
    ml = get_ml_service()
    features = request.model_dump()
    result = ml.predict_antibiotic(features)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Antimicrobial susceptibility model is not available. "
                "Ensure backend/models_store/antibiotic_susceptibility artifacts exist."
            ),
        )
    return AntibioticPredictionResponse(**result)
