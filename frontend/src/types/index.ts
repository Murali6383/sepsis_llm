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

export interface AntibioticFeaturesRequest {
  organism: string;
  antibiotic: string;
  culture_description?: string;
  ordering_mode?: string;
}

export interface AntibioticPredictionResponse {
  predicted_class: string;
  probabilities: Record<string, number>;
  model_type: string;
  model_version?: string;
  data_source?: string;
  organism?: string;
  antibiotic?: string;
  culture_description?: string;
  ordering_mode?: string;
  data_disclaimer: string;
  shap_available?: boolean;
  shap_values?: ShapValue[] | null;
  metrics_summary?: Record<string, unknown> | null;
}

export interface CandidateAntibioticRanking {
  antibiotic: string;
  rank: number;
  suitability_score: number;
  predicted_susceptibility: 'Susceptible' | 'Intermediate' | 'Resistant' | string;
  resistance_probability: number;
  susceptibility_probability: number;
  intermediate_probability: number;
  stewardship_tier: string;
  allergy_contraindication: boolean;
  allergy_warning?: string | null;
  renal_warning?: string | null;
  hepatic_warning?: string | null;
  interaction_warning?: string | null;
  warnings: string[];
  shap_values?: ShapValue[] | null;
}

export interface PatientAntibioticSuitabilityRequest {
  patient_age: number;
  patient_gender: string;
  infection_site: string;
  identified_organism: string;
  antibiotic_resistance_profile?: string[];
  previous_antibiotic_exposure?: string[];
  drug_allergies?: string[];
  creatinine: number;
  egfr: number;
  total_bilirubin?: number | null;
  ast_alt_elevated?: boolean;
  current_medications?: string[];
  candidate_antibiotics?: string[];
}

export interface PatientAntibioticSuitabilityResponse {
  identified_organism: string;
  infection_site: string;
  patient_summary: Record<string, unknown>;
  ranked_candidates: CandidateAntibioticRanking[];
  model_architecture: string;
  data_source: string;
  disclaimer: string;
  // Optional convenience aliases for report/explain
  predicted_class?: string;
  data_disclaimer?: string;
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
  antibiotic_result?:
    | AntibioticPredictionResponse
    | PatientAntibioticSuitabilityResponse
    | null;
}

export type RiskLevel = 'low' | 'moderate' | 'high';
