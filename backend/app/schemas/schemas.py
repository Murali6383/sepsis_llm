"""Pydantic schemas for all API request/response models."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ── Auth ──────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


# ── Sepsis Prediction ─────────────────────────────────────────────────────────

class SepsisFeaturesRequest(BaseModel):
    age: float = Field(..., ge=0, le=120)
    heart_rate: Optional[float] = Field(None, ge=0, le=300)
    temperature: Optional[float] = Field(None, ge=30, le=45)
    respiratory_rate: Optional[float] = Field(None, ge=0, le=60)
    spo2: Optional[float] = Field(None, ge=50, le=100)
    systolic_bp: Optional[float] = Field(None, ge=0, le=300)
    diastolic_bp: Optional[float] = Field(None, ge=0, le=200)
    map: Optional[float] = Field(None, ge=0, le=250)
    wbc: Optional[float] = Field(None, ge=0, le=100)
    lactate: Optional[float] = Field(None, ge=0, le=30)
    creatinine: Optional[float] = Field(None, ge=0, le=20)
    platelets: Optional[float] = Field(None, ge=0, le=2000)


class SepsisPredictionResponse(BaseModel):
    sepsis_probability: float
    sepsis_label: int
    threshold: float
    risk_level: str  # "low" | "moderate" | "high"
    model_type: str
    shap_available: bool
    shap_values: Optional[List[Dict[str, Any]]] = None


# ── Organ Risk ────────────────────────────────────────────────────────────────

class AKIFeaturesRequest(BaseModel):
    age: float = Field(..., ge=0, le=120)
    creatinine: Optional[float] = Field(None, ge=0, le=20)
    baseline_creatinine: Optional[float] = Field(None, ge=0, le=20)
    bun: Optional[float] = Field(None, ge=0, le=200)
    urine_output_ml: Optional[float] = Field(None, ge=0)
    heart_rate: Optional[float] = Field(None, ge=0, le=300)
    map: Optional[float] = Field(None, ge=0, le=250)


class CardioFeaturesRequest(BaseModel):
    age: float = Field(..., ge=0, le=120)
    heart_rate: Optional[float] = Field(None, ge=0, le=300)
    systolic_bp: Optional[float] = Field(None, ge=0, le=300)
    diastolic_bp: Optional[float] = Field(None, ge=0, le=200)
    map: Optional[float] = Field(None, ge=0, le=250)
    lactate: Optional[float] = Field(None, ge=0, le=30)
    vasopressor_use: Optional[int] = Field(None, ge=0, le=1)


class LiverFeaturesRequest(BaseModel):
    age: float = Field(..., ge=0, le=120)
    total_bilirubin: Optional[float] = Field(None, ge=0)
    direct_bilirubin: Optional[float] = Field(None, ge=0)
    ast: Optional[float] = Field(None, ge=0)
    alt: Optional[float] = Field(None, ge=0)
    alkaline_phosphatase: Optional[float] = Field(None, ge=0)
    albumin: Optional[float] = Field(None, ge=0, le=10)
    platelets: Optional[float] = Field(None, ge=0, le=2000)
    inr: Optional[float] = Field(None, ge=0, le=10)
    map: Optional[float] = Field(None, ge=0, le=250)


class NeuroFeaturesRequest(BaseModel):
    age: float = Field(..., ge=0, le=120)
    gcs_eye: Optional[float] = Field(None, ge=1, le=4)
    gcs_verbal: Optional[float] = Field(None, ge=1, le=5)
    gcs_motor: Optional[float] = Field(None, ge=1, le=6)
    total_gcs: Optional[float] = Field(None, ge=3, le=15)
    mental_status: Optional[str] = None
    sedation_status: Optional[str] = None
    map: Optional[float] = Field(None, ge=0, le=250)
    spo2: Optional[float] = Field(None, ge=50, le=100)
    oxygen_support: Optional[str] = None


class RespFeaturesRequest(BaseModel):
    age: float = Field(..., ge=0, le=120)
    respiratory_rate: Optional[float] = Field(None, ge=0, le=60)
    spo2: Optional[float] = Field(None, ge=50, le=100)
    oxygen_support: Optional[str] = None
    ventilation_status: Optional[str] = None
    heart_rate: Optional[float] = Field(None, ge=0, le=300)


class OrganRiskResponse(BaseModel):
    organ: str
    probability: float
    label: int
    threshold: float
    risk_level: str
    model_type: str
    shap_available: bool
    shap_values: Optional[List[Dict[str, Any]]] = None


# ── Antibiotic / Antimicrobial Susceptibility ──────────────────────────────────

class AntibioticFeaturesRequest(BaseModel):
    # Real ARMD-ECUH clinical fields
    organism: Optional[str] = Field(None, max_length=128)
    antibiotic: Optional[str] = Field(None, max_length=128)
    culture_description: Optional[str] = Field("URINE", max_length=64)
    ordering_mode: Optional[str] = Field("Inpatient", max_length=64)

    # Patient encounter & legacy demo fields (optional for backward compatibility)
    age: Optional[float] = Field(None, ge=0, le=120)
    gender: Optional[str] = Field(None, max_length=10)
    heart_rate: Optional[float] = Field(None, ge=0, le=300)
    systolic_bp: Optional[float] = Field(None, ge=0, le=300)
    diastolic_bp: Optional[float] = Field(None, ge=0, le=200)
    map: Optional[float] = Field(None, ge=0, le=250)
    respiratory_rate: Optional[float] = Field(None, ge=0, le=60)
    temperature: Optional[float] = Field(None, ge=30, le=45)
    spo2: Optional[float] = Field(None, ge=50, le=100)
    wbc: Optional[float] = Field(None, ge=0, le=100)
    lactate: Optional[float] = Field(None, ge=0, le=30)
    creatinine: Optional[float] = Field(None, ge=0, le=20)
    bilirubin: Optional[float] = Field(None, ge=0)
    platelets: Optional[float] = Field(None, ge=0, le=2000)
    glucose: Optional[float] = Field(None, ge=0, le=1000)
    crp: Optional[float] = Field(None, ge=0)
    urine_output: Optional[float] = Field(None, ge=0)
    gcs: Optional[int] = Field(None, ge=3, le=15)
    vasopressor: Optional[int] = Field(None, ge=0, le=1)
    mechanical_ventilation: Optional[int] = Field(None, ge=0, le=1)
    infection_source_synthetic: Optional[str] = None
    beta_lactam_allergy_synthetic: Optional[int] = Field(None, ge=0, le=1)
    egfr_computed: Optional[float] = Field(None, ge=0)
    renal_dose_adjustment_review: Optional[int] = Field(None, ge=0, le=1)


class AntibioticPredictionResponse(BaseModel):
    predicted_class: str
    probabilities: Dict[str, float]
    model_type: str = "LightGBM Multiclass Classifier"
    model_version: str = "v1.0-armd"
    data_source: str = "ARMD-ECUH (ECU Health, N=1,931,685 culture tests)"
    organism: Optional[str] = None
    antibiotic: Optional[str] = None
    culture_description: Optional[str] = None
    ordering_mode: Optional[str] = None
    shap_available: bool = False
    shap_values: Optional[List[Dict[str, Any]]] = None
    metrics_summary: Optional[Dict[str, Any]] = None
    data_disclaimer: str = (
        "Data source: ARMD-ECUH retrospective research dataset. "
        "Model outputs are research estimates and have not been prospectively or clinically validated. "
        "Susceptibility prediction is an in-vitro phenotype classification and does NOT constitute "
        "an individualized antibiotic recommendation."
    )


class PDFReportRequest(BaseModel):
    sepsis_result: Optional[Dict[str, Any]] = None
    organ_results: Optional[List[Dict[str, Any]]] = None
    antibiotic_result: Optional[Dict[str, Any]] = None
    shap_values: Optional[List[Dict[str, Any]]] = None
    patient_ref: Optional[str] = Field("PT-ANON-DEMO", max_length=64)
    input_features: Optional[Dict[str, Any]] = None


# ── AI / LLM ──────────────────────────────────────────────────────────────────

class ExplainRequest(BaseModel):
    sepsis_result: Optional[Dict[str, Any]] = None
    organ_results: Optional[List[Dict[str, Any]]] = None
    antibiotic_result: Optional[Dict[str, Any]] = None
    shap_values: Optional[List[Dict[str, Any]]] = None
    missing_features: Optional[List[str]] = None

    @field_validator("sepsis_result", "antibiotic_result", mode="before")
    @classmethod
    def _limit_dict_size(cls, v: Any) -> Any:
        if isinstance(v, dict) and len(str(v)) > 4000:
            raise ValueError("Payload too large.")
        return v


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    patient_context: Dict[str, Any] = Field(default_factory=dict)
    message: str = Field(..., min_length=1, max_length=1000)
    history: List[ChatMessage] = Field(default_factory=list, max_length=10)


class ChatResponse(BaseModel):
    answer: str
    source: str = "groq_llm"
    disclaimer: str = (
        "LLM-generated explanation — not a medical diagnosis or clinical recommendation."
    )


class ReportRequest(BaseModel):
    sepsis_result: Optional[Dict[str, Any]] = None
    organ_results: Optional[List[Dict[str, Any]]] = None
    antibiotic_result: Optional[Dict[str, Any]] = None
    shap_values: Optional[List[Dict[str, Any]]] = None
    patient_ref: Optional[str] = Field(None, max_length=64)  # de-identified ref only


class ReportResponse(BaseModel):
    report_text: str
    generated_at: str
    disclaimer: str = (
        "RESEARCH USE ONLY — Antimicrobial susceptibility predictions are derived from "
        "the retrospective ARMD-ECUH dataset (ECU Health); sepsis and organ risk models "
        "are research prototypes. This report does not constitute a medical diagnosis "
        "or individual treatment recommendation."
    )


class AIHealthResponse(BaseModel):
    llm_configured: bool
    llm_model: str
    groq_api_key_present: bool
    note: str


class ExplainResponse(BaseModel):
    explanation: str
    source: str = "groq_llm"
    disclaimer: str = (
        "LLM-generated explanation — not a medical diagnosis or clinical recommendation."
    )
