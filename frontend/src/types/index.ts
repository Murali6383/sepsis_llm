/**
 * TypeScript types for SepsisGuardian AI.
 * Mirrors backend Pydantic schemas exactly.
 */

// ── Auth ──────────────────────────────────────────────────────────────────────
export interface LoginRequest {
  username: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  role: string;
}

// ── ML Results ────────────────────────────────────────────────────────────────

export interface ShapValue {
  feature: string;
  shap_value: number;
  value: number | null;
}

export interface SepsisPredictionResponse {
  sepsis_probability: number;
  sepsis_label: number;
  threshold: number;
  risk_level: 'low' | 'moderate' | 'high';
  model_type: string;
  shap_available: boolean;
  shap_values: ShapValue[] | null;
}

export interface OrganRiskResponse {
  organ: string;
  probability: number;
  label: number;
  threshold: number;
  risk_level: 'low' | 'moderate' | 'high';
  model_type: string;
  shap_available: boolean;
  shap_values: ShapValue[] | null;
}

export interface AntibioticPredictionResponse {
  predicted_class: string;
  probabilities: Record<string, number>;
  model_type: string;
  data_disclaimer: string;
}

// ── AI/LLM ────────────────────────────────────────────────────────────────────

export interface AIHealthResponse {
  llm_configured: boolean;
  llm_model: string;
  groq_api_key_present: boolean;
  note: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatRequest {
  patient_context: Record<string, unknown>;
  message: string;
  history: ChatMessage[];
}

export interface ChatResponse {
  answer: string;
  source: string;
  disclaimer: string;
}

export interface ExplainRequest {
  sepsis_result?: Record<string, unknown>;
  organ_results?: Record<string, unknown>[];
  antibiotic_result?: Record<string, unknown>;
  shap_values?: ShapValue[];
  missing_features?: string[];
}

export interface ExplainResponse {
  explanation: string;
  source: string;
  disclaimer: string;
}

export interface ReportRequest {
  sepsis_result?: Record<string, unknown>;
  organ_results?: Record<string, unknown>[];
  antibiotic_result?: Record<string, unknown>;
  shap_values?: ShapValue[];
  patient_ref?: string;
}

export interface ReportResponse {
  report_text: string;
  generated_at: string;
  disclaimer: string;
}

// ── Patient context for UI state ─────────────────────────────────────────────
export interface PatientContext {
  sepsis_result?: SepsisPredictionResponse | null;
  organ_results?: OrganRiskResponse[];
  antibiotic_result?: AntibioticPredictionResponse | null;
}

export type RiskLevel = 'low' | 'moderate' | 'high';
