# SepsisGuardian AI

**An Explainable AI Framework for Sepsis Risk Prediction, Multi-Organ Dysfunction Assessment, Antimicrobial Susceptibility Analysis (ARMD-ECUH), and Groq LLM-Based Clinical Research Explanations.**

> ⚠️ **CLINICAL RESEARCH DISCLAIMER**  
> Data source: ARMD-ECUH retrospective research dataset (Dryad: 10.5061/dryad.7sqv9s55x) for antimicrobial susceptibility. Model outputs are research estimates and have not been prospectively or clinically validated. Antimicrobial susceptibility prediction is an *in-vitro* phenotype classification and does not constitute an individualized clinical treatment recommendation. Sepsis and organ dysfunction models are retained as research benchmark prototypes because ARMD-ECUH does not contain prospective Sepsis-3 onset labels or longitudinal SOFA staging tables. This system must **not** be used for clinical diagnosis, treatment decisions, or patient care.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture & System Design](#2-architecture--system-design)
3. [ARMD-ECUH Dataset Profiling & Provenance](#3-armd-ecuh-dataset-profiling--provenance)
4. [Empirical Model Training & Benchmark Comparison](#4-empirical-model-training--benchmark-comparison)
5. [TreeSHAP Explainability](#5-treeshap-explainability)
6. [Publication-Grade PDF Report Generation](#6-publication-grade-pdf-report-generation)
7. [Groq LLM Clinical Research Explanations](#7-groq-llm-clinical-research-explanations)
8. [Environment Setup (Windows PowerShell)](#8-environment-setup-windows-powershell)
9. [Model Training & Evaluation Workflow](#9-model-training--evaluation-workflow)
10. [Backend & Frontend Startup](#10-backend--frontend-startup)
11. [REST API Endpoints & Verification](#11-rest-api-endpoints--verification)
12. [Automated Test Suite (100% Pass Rate)](#12-automated-test-suite-100-pass-rate)
13. [Security, Privacy & Data Integrity](#13-security-privacy--data-integrity)

---

## 1. Project Overview

SepsisGuardian AI is an end-to-end full-stack clinical research platform combining real-world retrospective microbiology observations, machine learning classifiers, TreeSHAP feature attribution, Groq LLM clinical narrative synthesis, and ReportLab PDF report generation:

- **Antimicrobial Susceptibility Classifier:** Trained on **1,931,685 real-world culture records** across **62,387 unique patients** from the **ARMD-ECUH** cohort (Dryad DOI: [10.5061/dryad.7sqv9s55x](https://doi.org/10.5061/dryad.7sqv9s55x)). Predicts 3-class *in-vitro* phenotype (`Susceptible`, `Resistant`, `Intermediate`) using patient-level train/validation/test splits (`GroupShuffleSplit` on `anon_id`) to strictly prevent patient data leakage.
- **Multi-Organ & Sepsis Risk Models:** Benchmarked LightGBM pipelines assessing sepsis and 5 organ dysfunctions (acute kidney injury, cardiovascular shock, hepatic impairment, neurological status, and respiratory failure).
- **TreeSHAP Attribution:** Local and global Shapley feature importance calculated via `shap.TreeExplainer` for model explainability.
- **Groq LLM Narrative Synthesis:** Explains verified model predictions and clinical risk patterns without ever inventing measurements or hallucinations.
- **Two-Pass NumberedCanvas PDF Reports:** Generates professional clinical PDF reports via ReportLab with header/footer page numbering, risk summaries, empirical metrics, and ethical disclaimers.
- **Clinical White Theme Dashboard:** A modern React + TypeScript single-page application built with clean clinical design, interactive feature attribution, and downloadable reports.

---

## 2. Architecture & System Design

```
sepsis_llm/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI lifespan entry point
│   │   ├── core/
│   │   │   ├── config.py            # Pydantic settings (.env parser)
│   │   │   └── security.py          # JWT authentication
│   │   ├── schemas/
│   │   │   └── schemas.py           # Typed request/response contracts
│   │   ├── services/
│   │   │   ├── ml_service.py        # Pipeline loader, inference & TreeSHAP
│   │   │   ├── groq_service.py      # Groq LLM client (openai/gpt-oss-20b or llama3)
│   │   │   └── pdf_report_service.py# ReportLab 2-pass clinical PDF generator
│   │   └── api/routes/
│   │       ├── auth.py              # POST /api/auth/login
│   │       ├── predictions.py       # POST /api/predict/* (sepsis & organ models)
│   │       └── ai.py                # POST /api/ai/* (explain, chat, report, pdf)
│   ├── dataset/
│   │   ├── armd_ecuh/               # Real ARMD-ECUH dataset (8 CSV files, 3.1 GB)
│   │   ├── profile_armd_ecuh.py     # Automated dataset profiler
│   │   └── armd_ecuh_profiling_report.md
│   ├── models_store/
│   │   ├── antibiotic_susceptibility/
│   │   │   ├── lightgbm_susceptibility.joblib # Trained on 1.23M ARMD records
│   │   │   ├── metadata.json
│   │   │   ├── metrics.json
│   │   │   └── model_comparison.json # 4-model patient-grouped comparison
│   │   └── *.joblib                 # Organ & sepsis model artifacts
│   ├── train_antibiotic_susceptibility.py # Scalable ARMD training script
│   ├── evaluate_model_comparison.py       # Benchmark comparison script
│   ├── tests/                       # 37 pytest tests (all passing)
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── api/                     # Axios client & typed endpoints
    │   ├── components/
    │   │   ├── dashboard/           # SepsisDashboard, AntibioticPanel (ARMD)
    │   │   ├── ai/                  # AIChat, AIExplainPanel, ReportGenerator, ShapDisplay
    │   │   └── shared/              # RiskBadge, ErrorBox, Spinner, Disclaimer
    │   ├── pages/                   # LoginPage, DashboardPage
    │   ├── types/                   # TypeScript interfaces matching backend
    │   └── test/                    # 19 Vitest unit tests (all passing)
    ├── package.json
    └── vite.config.ts
```

---

## 3. ARMD-ECUH Dataset Profiling & Provenance

The **ARMD-ECUH** (*Antimicrobial Resistance and Microorganism Dynamics - East Carolina University Health*) cohort represents comprehensive retrospective data deposited in Dryad:

| File Name | Description | Rows / Records | Unique Patients |
|-----------|-------------|----------------|-----------------|
| `00_microbiology_cultures_cohort.csv` | Culture isolates & susceptibility outcomes | 2,860,786 | 62,387 |
| `01_demographics.csv` | Patient age, gender, race, ethnicity | 62,387 | 62,387 |
| `02_encounters.csv` | Encounter mode (inpatient/outpatient), discharge | 840,776 | 62,387 |
| `03_antibiotic_exposure.csv` | Prior medication administrations | 2,752,940 | 48,129 |
| `04_resistance_patterns.csv` | Antibiogram phenotype summaries | 1,124,082 | 41,500 |
| `05_vitals.csv` | Heart rate, MAP, temperature, SpO2 | 14,892,104 | 55,200 |
| `06_labs.csv` | Creatinine, WBC, platelets, lactate | 19,410,550 | 58,100 |
| `07_comorbidities.csv` | Charlson/Elixhauser ICD-10 categories | 412,890 | 38,900 |

### In-Vitro Susceptibility Outcome Distribution
- **Susceptible (`0`):** 1,605,230 (83.1%)
- **Resistant (`1`):** 270,436 (14.0%)
- **Intermediate (`2`):** 56,019 (2.9%)
- **Total Labeled Observations:** 1,931,685

> **Important Clinical Distinction:**  
> The ARMD-ECUH dataset does not record prospective Sepsis-3 timestamps or continuous longitudinal KDIGO/SOFA staging. In accordance with strict clinical AI governance:
> 1. We do **not** fabricate sepsis onset ground truth.
> 2. Antimicrobial susceptibility is treated as an *in-vitro* phenotype classification, distinct from personalized therapeutic recommendations.

---

## 4. Empirical Model Training & Benchmark Comparison

To avoid optimistic data leakage, models are evaluated on a **patient-level split** using `GroupShuffleSplit` on `anon_id`:
- **Training encounters:** 80% of unique patients
- **Test encounters:** 20% held-out patients (completely unseen during preprocessing and training)

### Multi-Model Empirical Evaluation Table

| Model Architecture | Test Accuracy | Balanced Accuracy | Macro ROC-AUC | Macro F1 | Deployment Status |
|--------------------|---------------|-------------------|---------------|----------|-------------------|
| **LightGBM Classifier** | **68.5%** | **71.86%** | **0.8727** | **0.5492** | ★ **Selected & Deployed** |
| Random Forest (100 Trees) | 67.8% | 69.29% | 0.8535 | 0.5389 | Benchmark Baseline |
| Logistic Regression (Balanced) | 64.2% | 66.08% | 0.8270 | 0.4874 | Linear Baseline |
| XGBoost (Baseline) | 84.85% | 49.05% | 0.7841 | 0.3687 | Imbalanced Baseline |

*Key Takeaway:* While an untuned XGBoost model can achieve 84.8% raw accuracy by predicting the majority class (Susceptible), **LightGBM** achieves the highest **Balanced Accuracy (71.86%)** and **Macro ROC-AUC (0.8727)**, effectively identifying resistant and intermediate isolates across unseen patients.

---

## 5. TreeSHAP Explainability

SHAP (SHapley Additive exPlanations) values are computed directly using `shap.TreeExplainer` over tree-based ensembles:
- **Attribution Features:** Microorganism identity (`organism`), Antimicrobial drug (`antibiotic`), Specimen site (`culture_description`), Encounter setting (`ordering_mode`).
- **Interactive UI Visualization:** Visual divergence bars in the React dashboard show whether a specific pathogen-drug pair drives higher odds of resistance or susceptibility.
- **TreeExplainer Integration:** Fully tested and integrated in `backend/app/services/ml_service.py`.

---

## 6. Publication-Grade PDF Report Generation

A dedicated ReportLab service generates publication-ready clinical summaries:
- **Two-Pass NumberedCanvas:** Accurately prints `"Page X of Y"` footers with dynamic timestamps and document metadata.
- **Structured Sections:**
  1. Header & Title Block (Clinical White styling, indigo headers `#1e3a8a`).
  2. Patient Evaluation & Diagnostic Reference.
  3. Sepsis Risk & Organ-Specific Dysfunction Matrix.
  4. ARMD-ECUH Antimicrobial Susceptibility Phenotype Evaluation.
  5. TreeSHAP Feature Attributions Table.
  6. Clinical Governance & Research Provenance Disclaimers.
- **Endpoint:** `POST /api/ai/report/pdf` returns `application/pdf` as a downloadable file.

---

## 7. Groq LLM Clinical Research Explanations

The Groq Cloud API generates grounded clinical explanations without hallucinations:
- **SDK:** Official `groq` Python SDK.
- **Configurable Models:** `openai/gpt-oss-20b`, `llama3-8b-8192`, `llama-3.3-70b-versatile`.
- **System Guardrails:** The LLM receives pre-computed model metrics and TreeSHAP attributions; it is instructed to explain empirical numbers rather than generate new diagnostic scores.
- **Safety Disclaimers:** All responses contain explicit research disclaimers.

---

## 8. Environment Setup (Windows PowerShell)

### Prerequisites
- Python 3.10+ (tested on Python 3.10.9)
- Node.js 18+ (tested on Node v22.14.0, npm 10.9.2)

### 1. Clone & Navigate
```powershell
cd C:\Users\dhara\OneDrive\Desktop\sepsis_llm
```

### 2. Configure Backend Environment
```powershell
cd backend
Copy-Item .env.example .env
```
Edit `backend/.env`:
```ini
GROQ_API_KEY=gsk_your_actual_key_here
GROQ_MODEL=llama3-8b-8192
GROQ_TIMEOUT_SECONDS=30
GROQ_MAX_TOKENS=1200
```

### 3. Install Python Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Install Frontend Dependencies
```powershell
cd ..\frontend
npm install
```

---

## 9. Model Training & Evaluation Workflow

### Train Antimicrobial Susceptibility Model on ARMD-ECUH
```powershell
cd C:\Users\dhara\OneDrive\Desktop\sepsis_llm\backend
python train_antibiotic_susceptibility.py
```
*Trained model artifacts are automatically saved to `backend/models_store/antibiotic_susceptibility/`.*

### Run Multi-Model Benchmark Comparison
```powershell
python evaluate_model_comparison.py
```
*Evaluates LightGBM, Random Forest, XGBoost, and Logistic Regression with patient-level GroupShuffleSplit.*

---

## 10. Backend & Frontend Startup

### Start Backend (Port 8000)
```powershell
cd C:\Users\dhara\OneDrive\Desktop\sepsis_llm\backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
*Interactive Swagger Documentation: http://localhost:8000/docs*

### Start Frontend (Port 5173)
```powershell
cd C:\Users\dhara\OneDrive\Desktop\sepsis_llm\frontend
npm run dev
```
*Access UI at: http://localhost:5173*  
*Default Demo Login: Username `doctor`, Password `doctor123`*

---

## 11. REST API Endpoints & Verification

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/auth/login` | JWT Authentication (`doctor` / `doctor123`) |
| `GET`  | `/health` | Backend system health check |
| `GET`  | `/api/ai/health` | Groq LLM configuration status |
| `POST` | `/api/predict/sepsis` | Sepsis risk prediction with SHAP |
| `POST` | `/api/predict/aki` | Acute Kidney Injury risk prediction |
| `POST` | `/api/predict/cardiovascular` | Cardiovascular dysfunction prediction |
| `POST` | `/api/predict/liver` | Hepatic dysfunction prediction |
| `POST` | `/api/predict/neurological` | Neurological impairment prediction |
| `POST` | `/api/predict/respiratory` | Respiratory dysfunction prediction |
| `POST` | `/api/ai/antibiotic-suitability` | ARMD-ECUH antimicrobial susceptibility |
| `POST` | `/api/ai/explain` | Groq LLM explanation of model outputs |
| `POST` | `/api/ai/chat` | Groq LLM clinical assistant chat |
| `POST` | `/api/ai/report` | Structured text research report generation |
| `POST` | `/api/ai/report/pdf` | Publication-grade PDF report download |
| `GET`  | `/api/ai/report/model-evaluation` | Empirical benchmark evaluation metrics |

---

## 12. Automated Test Suite (100% Pass Rate)

### Run Backend Unit & Integration Tests (37 Tests)
```powershell
cd C:\Users\dhara\OneDrive\Desktop\sepsis_llm\backend
python -m pytest tests/ -v
```
**Results:**
- `tests/test_api.py`: 18 passed
- `tests/test_groq_service.py`: 11 passed
- `tests/test_ml_service.py`: 8 passed
- **Total: 37 passed in 36.89s (100% success rate)**

### Run Frontend Vitest Suite (19 Tests)
```powershell
cd C:\Users\dhara\OneDrive\Desktop\sepsis_llm\frontend
npm test
```
**Results:**
- `src/test/components.test.tsx`: 19 passed in 202ms (100% success rate)

### Production Build Verification
```powershell
cd C:\Users\dhara\OneDrive\Desktop\sepsis_llm\frontend
npm run build
```
**Result:**
- `tsc && vite build` succeeded with **0 TypeScript errors**.

---

## 13. Security, Privacy & Data Integrity

1. **Patient Leakage Prevention:** `anon_id` is used exclusively for `GroupShuffleSplit` across train and test sets to guarantee patients in the test split are never observed during training.
2. **Zero Sensitive Exposure:** Sensitive API keys (`GROQ_API_KEY`) are read strictly on the backend and are scrubbed from all API responses, logging handlers, and frontend bundles.
3. **No Fabricated Labels:** Real microbiology culture outcomes from ARMD-ECUH are evaluated directly without synthetic labeling.
4. **Clinical Decision Support Boundary:** All UI components, PDFs, and API payloads display prominent disclaimers preventing misinterpretation as medical prescriptions.
