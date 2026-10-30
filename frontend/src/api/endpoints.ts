/**
 * API functions — typed wrappers over the Axios client.
 * Mirrors backend FastAPI endpoints exactly.
 */
import api from './client';
import type {
  LoginRequest,
  TokenResponse,
  SepsisPredictionResponse,
  OrganRiskResponse,
  AntibioticPredictionResponse,
  AIHealthResponse,
  ChatRequest,
  ChatResponse,
  ExplainRequest,
  ExplainResponse,
  ReportRequest,
  ReportResponse,
} from '../types';

// ── Auth ──────────────────────────────────────────────────────────────────────
export const login = (data: LoginRequest) =>
  api.post<TokenResponse>('/api/auth/login', data).then(r => r.data);

// ── Sepsis Prediction ─────────────────────────────────────────────────────────
export const predictSepsis = (features: Record<string, unknown>) =>
  api.post<SepsisPredictionResponse>('/api/predict/sepsis', features).then(r => r.data);

// ── Organ Risk ────────────────────────────────────────────────────────────────
export const predictAKI = (features: Record<string, unknown>) =>
  api.post<OrganRiskResponse>('/api/predict/aki', features).then(r => r.data);

export const predictCardio = (features: Record<string, unknown>) =>
  api.post<OrganRiskResponse>('/api/predict/cardiovascular', features).then(r => r.data);

export const predictLiver = (features: Record<string, unknown>) =>
  api.post<OrganRiskResponse>('/api/predict/liver', features).then(r => r.data);

export const predictNeuro = (features: Record<string, unknown>) =>
  api.post<OrganRiskResponse>('/api/predict/neurological', features).then(r => r.data);

export const predictResp = (features: Record<string, unknown>) =>
  api.post<OrganRiskResponse>('/api/predict/respiratory', features).then(r => r.data);

// ── AI / LLM ──────────────────────────────────────────────────────────────────
export const getAIHealth = () =>
  api.get<AIHealthResponse>('/api/ai/health').then(r => r.data);

export const explainPrediction = (data: ExplainRequest) =>
  api.post<ExplainResponse>('/api/ai/explain', data).then(r => r.data);

export const chatWithAI = (data: ChatRequest) =>
  api.post<ChatResponse>('/api/ai/chat', data).then(r => r.data);

export const generateReport = (data: ReportRequest) =>
  api.post<ReportResponse>('/api/ai/report', data).then(r => r.data);

// ── Antibiotic Suitability / ARMD-ECUH Susceptibility ─────────────────────────
export const predictAntibiotic = (features: Record<string, unknown>) =>
  api.post<AntibioticPredictionResponse>('/api/ai/antibiotic-suitability', features).then(r => r.data);

// ── PDF Report & Evaluation Metrics ───────────────────────────────────────────
export const downloadPDFReport = async (data: ReportRequest): Promise<Blob> => {
  const response = await api.post('/api/ai/report/pdf', data, {
    responseType: 'blob',
  });
  return response.data;
};

export const getModelEvaluationMetrics = () =>
  api.get<Record<string, unknown>>('/api/ai/report/model-evaluation').then(r => r.data);
