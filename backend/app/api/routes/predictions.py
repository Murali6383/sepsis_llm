"""Prediction routes — sepsis and organ risk."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.security import TokenPayload, get_current_user
from app.schemas.schemas import (
    AKIFeaturesRequest,
    CardioFeaturesRequest,
    LiverFeaturesRequest,
    NeuroFeaturesRequest,
    OrganRiskResponse,
    RespFeaturesRequest,
    SepsisFeaturesRequest,
    SepsisPredictionResponse,
)
from app.services.ml_service import get_ml_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/predict", tags=["Predictions"])


@router.post("/sepsis", response_model=SepsisPredictionResponse)
def predict_sepsis(
    request: SepsisFeaturesRequest,
    user: TokenPayload = Depends(get_current_user),
):
    """
    Predict sepsis probability using LightGBM model.
    Returns probability, label, risk level, and optional SHAP values.
    All data is synthetic — research demonstration only.
    """
    ml = get_ml_service()
    features = request.model_dump()
    result = ml.predict_sepsis(features, include_shap=True)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sepsis model is not available. Run training first: python -m app.models.train_models",
        )
    return SepsisPredictionResponse(**result)


@router.post("/aki", response_model=OrganRiskResponse)
def predict_aki(
    request: AKIFeaturesRequest,
    user: TokenPayload = Depends(get_current_user),
):
    """Predict acute kidney injury risk."""
    ml = get_ml_service()
    result = ml.predict_aki(request.model_dump(), include_shap=True)
    if result is None:
        raise HTTPException(status_code=503, detail="AKI model unavailable.")
    return OrganRiskResponse(**result)


@router.post("/cardiovascular", response_model=OrganRiskResponse)
def predict_cardio(
    request: CardioFeaturesRequest,
    user: TokenPayload = Depends(get_current_user),
):
    """Predict cardiovascular risk."""
    ml = get_ml_service()
    result = ml.predict_cardio(request.model_dump(), include_shap=True)
    if result is None:
        raise HTTPException(status_code=503, detail="Cardiovascular model unavailable.")
    return OrganRiskResponse(**result)


@router.post("/liver", response_model=OrganRiskResponse)
def predict_liver(
    request: LiverFeaturesRequest,
    user: TokenPayload = Depends(get_current_user),
):
    """Predict liver risk."""
    ml = get_ml_service()
    result = ml.predict_liver(request.model_dump(), include_shap=True)
    if result is None:
        raise HTTPException(status_code=503, detail="Liver model unavailable.")
    return OrganRiskResponse(**result)


@router.post("/neurological", response_model=OrganRiskResponse)
def predict_neuro(
    request: NeuroFeaturesRequest,
    user: TokenPayload = Depends(get_current_user),
):
    """Predict neurological risk."""
    ml = get_ml_service()
    result = ml.predict_neuro(request.model_dump(), include_shap=True)
    if result is None:
        raise HTTPException(status_code=503, detail="Neurological model unavailable.")
    return OrganRiskResponse(**result)


@router.post("/respiratory", response_model=OrganRiskResponse)
def predict_resp(
    request: RespFeaturesRequest,
    user: TokenPayload = Depends(get_current_user),
):
    """Predict respiratory risk."""
    ml = get_ml_service()
    result = ml.predict_resp(request.model_dump(), include_shap=True)
    if result is None:
        raise HTTPException(status_code=503, detail="Respiratory model unavailable.")
    return OrganRiskResponse(**result)


@router.get("/models/status")
def models_status(user: TokenPayload = Depends(get_current_user)):
    """Return availability status of all loaded ML models."""
    ml = get_ml_service()
    return {"models_available": ml.models_available()}
