# SepsisGuardian AI

**An Explainable AI Framework for Sepsis Risk Prediction, Multi-Organ Dysfunction Assessment, Antibiotic Susceptibility Analysis, and Groq LLM-Based Clinical Research Explanations.**

> ⚠️ **RESEARCH DEMONSTRATION ONLY.** All datasets are fully synthetic. All model outputs are for research purposes. This system must **not** be used for clinical diagnosis, treatment decisions, or patient care.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Dataset Details](#3-dataset-details)
4. [Model Training Results](#4-model-training-results)
5. [Environment Setup (Windows)](#5-environment-setup-windows)
6. [Groq API Configuration](#6-groq-api-configuration)
7. [Backend Startup](#7-backend-startup)
8. [Frontend Startup](#8-frontend-startup)
9. [API Endpoints & Example Requests](#9-api-endpoints--example-requests)
10. [LLM Assistant Usage](#10-llm-assistant-usage)
11. [SHAP Explainability](#11-shap-explainability)
12. [Research Report Generation](#12-research-report-generation)
13. [Testing](#13-testing)
14. [Security & Privacy](#14-security--privacy)
15. [Known Issues & Limitations](#15-known-issues--limitations)
16. [Research-Only Disclaimer](#16-research-only-disclaimer)

---

## 1. Project Overview

SepsisGuardian AI is a full-stack explainable AI research platform that:

- Predicts **sepsis risk** using a trained LightGBM model
- Predicts **organ-specific risks** (AKI, cardiovascular, liver, neurological, respiratory) using five independent LightGBM models
- Predicts **antibiotic suitability** (demo) using a separate LightGBM multiclass model
- Generates **SHAP-based explanations** for model predictions
- Uses **Groq LLM** (cloud) to explain model outputs in plain English
- Provides an **AI research chatbot** grounded in actual model results
- Generates **structured research reports** from model outputs
- Presents everything in a **React + TypeScript dashboard**

---

## 2. Architecture

```
sepsis_llm/
├── synthetic_*.csv          # All synthetic datasets (project root)
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI entry point
│   │   ├── core/
│   │   │   ├── config.py    # Pydantic settings (reads .env)
│   │   │   └── security.py  # JWT auth (demo)
│   │   ├── schemas/
│   │   │   └── schemas.py   # Pydantic request/response models
│   │   ├── models/
│   │   │   └── train_models.py  # Training pipeline (all 7 models)
│   │   ├── services/
│   │   │   ├── ml_service.py    # Model loading + inference + SHAP
│   │   │   └── groq_service.py  # Groq LLM client + safe prompting
│   │   └── api/routes/
│   │       ├── auth.py          # POST /api/auth/login
│   │       ├── predictions.py   # POST /api/predict/*
│   │       └── ai.py            # POST /api/ai/*
│   ├── models_store/        # Saved .joblib model files (gitignored)
│   ├── tests/               # pytest tests (36 tests, all passing)
│   ├── requirements.txt
│   ├── pytest.ini
│   └── .env.example
└── frontend/
    ├── src/
    │   ├── api/             # Axios client + typed endpoint functions
    │   ├── components/      # React components (AI, dashboard, shared)
    │   ├── hooks/           # useAuth hook
    │   ├── pages/           # LoginPage, DashboardPage
    │   ├── types/           # TypeScript interfaces
    │   └── test/            # Vitest frontend tests
    ├── package.json
    └── vite.config.ts
```

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `ml_service.py` | Load models, run inference, compute SHAP |
| `groq_service.py` | Groq API client — explains ML results, never replaces them |
| `predictions.py` | REST endpoints for sepsis + organ risk models |
| `ai.py` | REST endpoints for explain, chat, report, health, antibiotic |
| `SepsisDashboard.tsx` | Prediction UI + SHAP display |
| `AntibioticPanel.tsx` | Separate antibiotic prediction UI |
| `AIChat.tsx` | LLM chat grounded in model outputs |
| `AIExplainPanel.tsx` | One-click LLM explanation of current results |
| `ReportGenerator.tsx` | LLM research report with copy + download |

---

## 3. Dataset Details

All datasets are in the **project root** (`sepsis_llm/`). Do not move or modify original CSV files.

### synthetic_sepsis_training.csv
- **Shape:** 5,000 × 15
- **Target:** `sepsis_label` (0: 2500, 1: 2500) — perfectly balanced
- **Features used:** age, heart_rate, temperature, respiratory_rate, spo2, systolic_bp, diastolic_bp, map, wbc, lactate, creatinine, platelets
- **Excluded:** patient_id, timestamp

### synthetic_aki_training.csv
- **Shape:** 5,000 × 10
- **Target:** `aki_label` (0: 3470, 1: 1530)
- **Features used:** age, creatinine, baseline_creatinine, bun, urine_output_ml, heart_rate, map
- **Missing:** bun (272), urine_output_ml (1069) — imputed with median

### synthetic_cardiovascular_risk_training.csv
- **Shape:** 5,000 × 10
- **Target:** `cardiovascular_risk_label` (0: 4443, 1: 557)
- **Features used:** age, heart_rate, systolic_bp, diastolic_bp, map, lactate, vasopressor_use

### synthetic_liver_risk_training.csv
- **Shape:** 5,000 × 13
- **Target:** `liver_risk_label` (0: 4122, 1: 878)
- **Features used:** age, total_bilirubin, direct_bilirubin, ast, alt, alkaline_phosphatase, albumin, platelets, inr, map

### synthetic_neurological_risk_training.csv
- **Shape:** 5,000 × 13
- **Target:** `neurological_risk_label` (0: 3866, 1: 1134)
- **Features used:** age, gcs_eye, gcs_verbal, gcs_motor, total_gcs, mental_status, sedation_status, map, spo2, oxygen_support

### synthetic_respiratory_risk_training.csv
- **Shape:** 5,000 × 9
- **Target:** `respiratory_risk_label` (0: 3567, 1: 1433)
- **Features used:** age, respiratory_rate, spo2, oxygen_support, ventilation_status, heart_rate

### synthetic_antibiotic.csv
- **Shape:** 5,000 × 29
- **Target:** `recommended_antibiotic_demo_label` — 10-class multiclass
- **label_origin:** `rule_table_demo_not_clinical_ground_truth` — **NOT clinical ground truth**
- **Features used (numerical):** age, heart_rate, systolic_bp, diastolic_bp, map, respiratory_rate, temperature, spo2, wbc, lactate, creatinine, bilirubin, platelets, glucose, crp, urine_output, gcs, vasopressor, mechanical_ventilation, beta_lactam_allergy_synthetic, egfr_computed, renal_dose_adjustment_review
- **Features used (categorical):** gender, infection_source_synthetic
- **Excluded (leakage):** patient_id, hour, data_origin, label_origin

---

## 4. Model Training Results

> ⚠️ Near-perfect scores reflect **rule-based synthetic labels**, not real-world performance. Do not interpret these as clinical validation.

| Model | ROC-AUC | Accuracy | Notes |
|-------|---------|----------|-------|
| Sepsis | 0.9999 | 99.7% | Balanced classes |
| AKI | 1.0000 | ~100% | Imbalanced (31% positive) |
| Cardiovascular | 1.0000 | ~100% | Highly imbalanced (11% positive) |
| Liver | 0.9998 | ~100% | Imbalanced (17% positive) |
| Neurological | 1.0000 | ~100% | Imbalanced (23% positive) |
| Respiratory | 1.0000 | ~100% | Imbalanced (29% positive) |
| Antibiotic | N/A (multiclass) | ~100% | 10-class, rule-based labels |

All models use: **LightGBM + sklearn Pipeline** (SimpleImputer → StandardScaler/OneHotEncoder → LGBMClassifier)

---

## 5. Environment Setup (Windows)

### Prerequisites
- Python 3.10+ (tested on 3.10.9)
- Node.js 18+ (tested on 22.14.0)

### Step 1 — Navigate to project

```powershell
cd C:\Users\dhara\OneDrive\Desktop\sepsis_llm
```

### Step 2 — Backend: Create virtual environment (optional but recommended)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3 — Install backend dependencies

```powershell
cd backend
pip install -r requirements.txt
```

### Step 4 — Configure backend .env

```powershell
Copy-Item .env.example .env
```

Then edit `backend/.env` and set your Groq API key (see Section 6).

### Step 5 — Train ML models (required before starting the backend)

```powershell
# From backend/ directory
python -m app.models.train_models
```

Models are saved to `backend/models_store/`. Training takes ~30 seconds.

### Step 6 — Start the FastAPI backend

```powershell
# From backend/ directory
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: http://localhost:8000/docs

### Step 7 — Install frontend dependencies

```powershell
cd ..\frontend
npm install
```

### Step 8 — Start the React frontend

```powershell
# From frontend/ directory
npm run dev
```

App available at: http://localhost:5173

### Step 9 — Run backend tests

```powershell
cd ..\backend
python -m pytest tests/ -v --tb=short
```

### Step 10 — Run frontend tests

```powershell
cd ..\frontend
npm test
```

### Step 11 — Build frontend for production

```powershell
npm run build
```

---

## 6. Groq API Configuration

> ⚠️ **Never commit your API key** to Git or share it in chat.

### Getting a Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign in and navigate to **API Keys**
3. Create a new key

### Setting the Key (privately)

Edit `backend/.env`:

```
GROQ_API_KEY=gsk_your_actual_key_here
GROQ_MODEL=llama3-8b-8192
GROQ_TIMEOUT_SECONDS=30
GROQ_MAX_TOKENS=1200
```

### Available Models

To list currently available Groq models:

```powershell
python -c "
import os; from groq import Groq
c = Groq(api_key=os.environ['GROQ_API_KEY'])
print([m.id for m in c.models.list().data])
"
```

Common available models (verify with your account):
- `llama3-8b-8192` (default — fast, good quality)
- `llama3-70b-8192` (higher quality, slower)
- `mixtral-8x7b-32768` (longer context)

### If the Configured Model Is Unavailable

The backend will return a clear error:
```json
{"detail": "Groq API authentication or validation error. Check your GROQ_API_KEY and GROQ_MODEL settings."}
```

Change `GROQ_MODEL` in `backend/.env` to an available model and restart the backend.

### What Data Is Sent to Groq

Only **de-identified model output summaries** are sent — no patient names, IDs, or raw clinical records. Example of what is sent:

```
SEPSIS MODEL:
  Probability: 0.82
  Risk Level: high
  Model Type: LightGBM

TOP SHAP FEATURES:
  ↑ lactate: SHAP=0.35, value=2.5
  ↓ map: SHAP=-0.22, value=68
```

---

## 7. Backend Startup

```powershell
cd C:\Users\dhara\OneDrive\Desktop\sepsis_llm\backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Port:** 8000 (configurable via `APP_PORT` in `.env`)
- **Docs:** http://localhost:8000/docs (Swagger UI)
- **ReDoc:** http://localhost:8000/redoc
- **Health:** http://localhost:8000/health

### Demo Login Credentials

| Username | Password | Role |
|----------|----------|------|
| doctor | doctor123 | doctor |
| nurse | nurse123 | nurse |
| admin | admin123 | admin |

---

## 8. Frontend Startup

```powershell
cd C:\Users\dhara\OneDrive\Desktop\sepsis_llm\frontend
npm run dev
```

- **Port:** 5173
- **URL:** http://localhost:5173
- Vite proxy forwards `/api/*` → `http://localhost:8000`

---

## 9. API Endpoints & Example Requests

All prediction endpoints require a Bearer JWT token obtained from `/api/auth/login`.

### Authentication

```bash
POST /api/auth/login
{
  "username": "doctor",
  "password": "doctor123"
}
```

### Sepsis Prediction

```bash
POST /api/predict/sepsis
Authorization: Bearer <token>
{
  "age": 65, "heart_rate": 110, "temperature": 38.5,
  "respiratory_rate": 22, "spo2": 95, "systolic_bp": 90,
  "diastolic_bp": 60, "map": 70, "wbc": 15.0,
  "lactate": 2.5, "creatinine": 1.8, "platelets": 100
}
```

Response includes: `sepsis_probability`, `sepsis_label`, `risk_level`, `shap_values`

### Organ Risk Predictions

```bash
POST /api/predict/aki         # AKI features
POST /api/predict/cardiovascular
POST /api/predict/liver
POST /api/predict/neurological
POST /api/predict/respiratory
```

### Antibiotic Suitability (DEMO — Synthetic Only)

```bash
POST /api/ai/antibiotic-suitability
Authorization: Bearer <token>
{
  "age": 55, "gender": "M", "gcs": 15,
  "vasopressor": 0, "mechanical_ventilation": 0,
  "infection_source_synthetic": "urinary",
  "beta_lactam_allergy_synthetic": 0,
  "renal_dose_adjustment_review": 0,
  "heart_rate": 100, "systolic_bp": 120, ...
}
```

### AI Explanation

```bash
POST /api/ai/explain
Authorization: Bearer <token>
{
  "sepsis_result": {"sepsis_probability": 0.82, "sepsis_label": 1, "risk_level": "high"},
  "organ_results": [{"organ": "acute_kidney_injury", "probability": 0.71, ...}],
  "shap_values": [{"feature": "lactate", "shap_value": 0.35, "value": 2.5}]
}
```

### AI Chat

```bash
POST /api/ai/chat
Authorization: Bearer <token>
{
  "patient_context": {"sepsis_result": {...}},
  "message": "Explain the current sepsis prediction.",
  "history": []
}
```

### Research Report

```bash
POST /api/ai/report
Authorization: Bearer <token>
{
  "sepsis_result": {...},
  "organ_results": [...],
  "patient_ref": "PT-ANON-001"
}
```

### AI Health Check

```bash
GET /api/ai/health
```

Response (no auth required):
```json
{
  "llm_configured": true,
  "llm_model": "llama3-8b-8192",
  "groq_api_key_present": true,
  "note": "LLM is ready."
}
```

---

## 10. LLM Assistant Usage

The AI assistant:
- **Explains** what the sepsis probability means statistically
- **Summarizes** which SHAP features influenced a prediction
- **Describes** organ-risk model outputs (when available)
- **Clarifies** the difference between sepsis risk and antibiotic suitability
- **Notes** missing data and model limitations

The assistant **never**:
- Invents risk scores or SHAP values
- Infers that an organ is safe when no model output exists
- Prescribes antibiotics or recommends treatment
- Claims synthetic results are clinically validated

---

## 11. SHAP Explainability

SHAP (SHapley Additive exPlanations) values are computed using `shap.TreeExplainer` for all LightGBM models.

- **Positive SHAP value** → feature increases predicted risk
- **Negative SHAP value** → feature decreases predicted risk
- **Magnitude** → relative importance for this specific prediction

SHAP values reflect statistical associations in the synthetic training data and must not be interpreted as proof of clinical causation.

SHAP computation may fail gracefully (returns `shap_available: false`) if the model type changes — this is handled without breaking the prediction.

---

## 12. Research Report Generation

Click **Generate Research Report** in the dashboard after running predictions.

The report includes:
1. Title and timestamp
2. De-identified patient reference (`PT-ANON-DEMO`)
3. Input data summary
4. Sepsis model output
5. Available organ-risk outputs
6. Antibiotic model result (if run)
7. SHAP explanation summary
8. Missing data and limitations
9. Research-only disclaimer

Reports can be **copied to clipboard** or **downloaded as .txt**.

---

## 13. Testing

### Backend Tests (pytest)

```powershell
cd backend
python -m pytest tests/ -v --tb=short
```

**Result: 36/36 passing** (verified)

Test coverage includes:
- Groq service configuration and missing key handling
- Mocked Groq successful responses
- Groq retry logic and permanent error handling (401)
- Invalid request validation (422)
- Antibiotic model loading and prediction
- SHAP value computation for sepsis model
- Sepsis score independence from antibiotic score
- API authentication and authorization
- Security: GROQ_API_KEY not exposed in any response
- Report generation with mocked Groq

### Frontend Tests (Vitest)

```powershell
cd frontend
npm test
```

Test coverage includes:
- Shared components (RiskBadge, ErrorBox, Disclaimer)
- ShapDisplay rendering and empty state
- AIChat: unconfigured state, send message, error handling
- AIExplainPanel: unconfigured, explanation display, error
- ReportGenerator: unconfigured, report display, antibiotic separation

### Frontend Build

```powershell
cd frontend
npm run build
```

---

## 14. Security & Privacy

- **GROQ_API_KEY** is read exclusively from `backend/.env` — never exposed in frontend source, API responses, logs, or browser network payloads
- JWT tokens are signed with `SECRET_KEY` (change in production)
- All inputs are validated by Pydantic schemas with field-level constraints
- Request size is limited (32 KB max body)
- Error responses never expose stack traces
- Only de-identified model output summaries are sent to Groq (no patient IDs)
- `.env` and model artifacts are gitignored

### Rate Limiting

No rate limiter is configured by default (this is a research demo). In production, add `slowapi` or a reverse proxy (nginx/Caddy) with rate limiting.

---

## 15. Known Issues & Limitations

| Issue | Status | Notes |
|-------|--------|-------|
| Perfect training scores (ROC~1.0) | Expected | Synthetic datasets have rule-based labels — not real clinical complexity |
| No rate limiting | Known | Add `slowapi` for production |
| Demo JWT auth | Known | Replace with real IdP (OAuth2/OIDC) in production |
| `on_event` deprecation warning | Minor | Replace with `lifespan` context manager in FastAPI |
| RAG not implemented | Documented | Would require sentence-transformers + vector DB; out of scope for this demo |
| Digital Twin | Not implemented | No "what-if scenario" panel — avoided per specification |
| asyncio_mode warning in pytest | Minor | Remove `asyncio_mode` from pytest.ini (no async tests currently) |

---

## 16. Research-Only Disclaimer

> This platform and all its models are built on **fully synthetic, rule-based datasets**. The `label_origin` field in the antibiotic dataset explicitly states `rule_table_demo_not_clinical_ground_truth`.
>
> **No model in this project has been:**
> - Trained on real patient data
> - Validated in a clinical setting
> - Approved for any medical use
> - Reviewed by clinical experts
>
> **This system must not be used for:**
> - Patient diagnosis or triage
> - Antibiotic prescribing decisions
> - Clinical risk stratification
> - Any patient-facing application
>
> LLM-generated explanations are produced by a cloud AI (Groq) and represent statistical summaries, not medical opinions. Always consult qualified healthcare professionals for clinical decisions.
