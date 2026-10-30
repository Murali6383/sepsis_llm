
"""
Prediction routes for SepsisGuardian AI.
Provides sepsis, organ-risk, full prediction and Digital Twin simulation APIs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

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
    FullPredictionResponse,
    SimulationRequest,
)

from app.services.ml_service import get_ml_service


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/predict",
    tags=["Predictions"],
)


# ---------------------------------------------------------
# Helper function
# ---------------------------------------------------------

def _check_prediction(result: Any, model_name: str) -> Dict[str, Any]:
    """Validate that an ML prediction is available."""

    if result is None:
        logger.error("%s model prediction unavailable", model_name)

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{model_name} model is unavailable. Please check model artifacts.",
        )

    if not isinstance(result, dict):
        logger.error(
            "%s returned an unexpected result type: %s",
            model_name,
            type(result).__name__,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{model_name} model returned an invalid prediction.",
        )

    return result


# ---------------------------------------------------------
# 1. Sepsis prediction
# ---------------------------------------------------------

@router.post(
    "/sepsis",
    response_model=SepsisPredictionResponse,
)
def predict_sepsis(
    request: SepsisFeaturesRequest,
    user: TokenPayload = Depends(get_current_user),
):
    """Predict sepsis probability using the LightGBM model."""

    ml = get_ml_service()

    result = ml.predict_sepsis(
        request.model_dump(),
        include_shap=True,
    )

    result = _check_prediction(result, "Sepsis")

    return SepsisPredictionResponse(**result)


# ---------------------------------------------------------
# 2. Acute Kidney Injury prediction
# ---------------------------------------------------------

@router.post(
    "/aki",
    response_model=OrganRiskResponse,
)
def predict_aki(
    request: AKIFeaturesRequest,
    user: TokenPayload = Depends(get_current_user),
):
    """Predict acute kidney injury risk."""

    ml = get_ml_service()

    result = ml.predict_aki(
        request.model_dump(),
        include_shap=True,
    )

    result = _check_prediction(result, "AKI")

    return OrganRiskResponse(**result)


# ---------------------------------------------------------
# 3. Cardiovascular prediction
# ---------------------------------------------------------

@router.post(
    "/cardiovascular",
    response_model=OrganRiskResponse,
)
def predict_cardio(
    request: CardioFeaturesRequest,
    user: TokenPayload = Depends(get_current_user),
):
    """Predict cardiovascular organ risk."""

    ml = get_ml_service()

    result = ml.predict_cardio(
        request.model_dump(),
        include_shap=True,
    )

    result = _check_prediction(result, "Cardiovascular")

    return OrganRiskResponse(**result)


# ---------------------------------------------------------
# 4. Liver prediction
# ---------------------------------------------------------

@router.post(
    "/liver",
    response_model=OrganRiskResponse,
)
def predict_liver(
    request: LiverFeaturesRequest,
    user: TokenPayload = Depends(get_current_user),
):
    """Predict liver dysfunction risk."""

    ml = get_ml_service()

    result = ml.predict_liver(
        request.model_dump(),
        include_shap=True,
    )

    result = _check_prediction(result, "Liver")

    return OrganRiskResponse(**result)


# ---------------------------------------------------------
# 5. Neurological prediction
# ---------------------------------------------------------

@router.post(
    "/neurological",
    response_model=OrganRiskResponse,
)
def predict_neuro(
    request: NeuroFeaturesRequest,
    user: TokenPayload = Depends(get_current_user),
):
    """Predict neurological organ risk."""

    ml = get_ml_service()

    result = ml.predict_neuro(
        request.model_dump(),
        include_shap=True,
    )

    result = _check_prediction(result, "Neurological")

    return OrganRiskResponse(**result)


# ---------------------------------------------------------
# 6. Respiratory prediction
# ---------------------------------------------------------

@router.post(
    "/respiratory",
    response_model=OrganRiskResponse,
)
def predict_resp(
    request: RespFeaturesRequest,
    user: TokenPayload = Depends(get_current_user),
):
    """Predict respiratory organ risk."""

    ml = get_ml_service()

    result = ml.predict_resp(
        request.model_dump(),
        include_shap=True,
    )

    result = _check_prediction(result, "Respiratory")

    return OrganRiskResponse(**result)


# ---------------------------------------------------------
# 7. Full prediction - Digital Twin
# ---------------------------------------------------------

@router.post(
    "/full",
    response_model=FullPredictionResponse,
)
def predict_full(
    sepsis: SepsisFeaturesRequest,
    aki: AKIFeaturesRequest,
    cardio: CardioFeaturesRequest,
    liver: LiverFeaturesRequest,
    neuro: NeuroFeaturesRequest,
    resp: RespFeaturesRequest,
    user: TokenPayload = Depends(get_current_user),
):
    """
    Generate sepsis and multi-organ risk predictions
    in a single API request.

    Includes:
    - Sepsis prediction
    - Kidney risk
    - Cardiovascular risk
    - Liver risk
    - Neurological risk
    - Respiratory risk
    - SHAP feature explanations
    """

    ml = get_ml_service()

    # Sepsis prediction
    sepsis_result = ml.predict_sepsis(
        sepsis.model_dump(),
        include_shap=True,
    )

    sepsis_result = _check_prediction(
        sepsis_result,
        "Sepsis",
    )

    # Organ predictions
    organ_results = {
        "aki": ml.predict_aki(
            aki.model_dump(),
            include_shap=True,
        ),

        "cardio": ml.predict_cardio(
            cardio.model_dump(),
            include_shap=True,
        ),

        "liver": ml.predict_liver(
            liver.model_dump(),
            include_shap=True,
        ),

        "neuro": ml.predict_neuro(
            neuro.model_dump(),
            include_shap=True,
        ),

        "resp": ml.predict_resp(
            resp.model_dump(),
            include_shap=True,
        ),
    }

    # Validate all organ predictions
    for organ_name, result in organ_results.items():
        organ_results[organ_name] = _check_prediction(
            result,
            organ_name.upper(),
        )

    # Build response
    return FullPredictionResponse(
        sepsis=SepsisPredictionResponse(**sepsis_result),

        organs={
            organ_name: OrganRiskResponse(**result)
            for organ_name, result in organ_results.items()
        },
    )


# ---------------------------------------------------------
# 8. Digital Twin What-if Simulation
# ---------------------------------------------------------

@router.post(
    "/simulate",
    response_model=FullPredictionResponse,
)
def simulate_prediction(
    request: SimulationRequest,
    user: TokenPayload = Depends(get_current_user),
):
    """
    Perform what-if simulation using modified patient inputs.

    Returns the same response structure as /full.
    """

    return predict_full(
        sepsis=request.sepsis,
        aki=request.aki,
        cardio=request.cardio,
        liver=request.liver,
        neuro=request.neuro,
        resp=request.resp,
        user=user,
    )


# ---------------------------------------------------------
# 9. ML Model Availability Status
# ---------------------------------------------------------

@router.get("/models/status")
def models_status(
    user: TokenPayload = Depends(get_current_user),
):
    """Return the availability of all loaded ML models."""

    ml = get_ml_service()

    try:
        available_models = ml.models_available()

        return {
            "status": "success",
            "models_available": available_models,
        }

    except Exception as exc:
        logger.exception("Unable to retrieve ML model status")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to retrieve ML model availability.",
        ) from exc
