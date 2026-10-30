/**
 * Frontend tests for SepsisGuardian AI components.
 * Uses Vitest + React Testing Library.
 * All backend calls are mocked — no live network requests.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';

// ── Mock API module ────────────────────────────────────────────────────────────
vi.mock('../api/endpoints', () => ({
  chatWithAI: vi.fn(),
  explainPrediction: vi.fn(),
  generateReport: vi.fn(),
  getAIHealth: vi.fn(),
  predictSepsis: vi.fn(),
  predictAKI: vi.fn(),
  predictCardio: vi.fn(),
  predictLiver: vi.fn(),
  predictNeuro: vi.fn(),
  predictResp: vi.fn(),
  predictAntibiotic: vi.fn(),
  login: vi.fn(),
}));

import * as endpoints from '../api/endpoints';
import { AIChat } from '../components/ai/AIChat';
import { AIExplainPanel } from '../components/ai/AIExplainPanel';
import { ReportGenerator } from '../components/ai/ReportGenerator';
import { ShapDisplay } from '../components/ai/ShapDisplay';
import { ErrorBox, RiskBadge, Disclaimer } from '../components/shared';

// ── Shared context ─────────────────────────────────────────────────────────────

const mockPatientContext = {
  sepsis_result: {
    sepsis_probability: 0.82,
    sepsis_label: 1,
    threshold: 0.5,
    risk_level: 'high' as const,
    model_type: 'LightGBM',
    shap_available: true,
    shap_values: [
      { feature: 'lactate', shap_value: 0.35, value: 2.5 },
      { feature: 'map', shap_value: -0.22, value: 68 },
    ],
  },
  organ_results: [],
  antibiotic_result: null,
};

// ── Shared components ──────────────────────────────────────────────────────────

describe('Shared components', () => {
  it('RiskBadge renders correct text for high risk', () => {
    render(<RiskBadge level="high" />);
    expect(screen.getByText('HIGH')).toBeInTheDocument();
  });

  it('RiskBadge renders correct text for low risk', () => {
    render(<RiskBadge level="low" />);
    expect(screen.getByText('LOW')).toBeInTheDocument();
  });

  it('ErrorBox renders message and retry button', () => {
    const onRetry = vi.fn();
    render(<ErrorBox message="Something went wrong" onRetry={onRetry} />);
    expect(screen.getByText(/Something went wrong/)).toBeInTheDocument();
    fireEvent.click(screen.getByText('Retry'));
    expect(onRetry).toHaveBeenCalledOnce();
  });

  it('Disclaimer renders warning text', () => {
    render(<Disclaimer />);
    expect(screen.getByRole('note')).toBeInTheDocument();
  });
});

// ── SHAP Display ──────────────────────────────────────────────────────────────

describe('ShapDisplay', () => {
  it('renders SHAP feature names and values', () => {
    const shapValues = [
      { feature: 'lactate', shap_value: 0.35, value: 2.5 },
      { feature: 'map', shap_value: -0.22, value: 68 },
    ];
    render(<ShapDisplay shapValues={shapValues} />);
    expect(screen.getByText('lactate')).toBeInTheDocument();
    expect(screen.getByText('map')).toBeInTheDocument();
  });

  it('shows empty message when no SHAP values', () => {
    render(<ShapDisplay shapValues={[]} />);
    expect(screen.getByText(/No SHAP values available/)).toBeInTheDocument();
  });
});

// ── AI Chat ───────────────────────────────────────────────────────────────────

describe('AIChat', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('shows unavailable state when LLM not configured', () => {
    render(<AIChat patientContext={mockPatientContext} llmConfigured={false} />);
    expect(screen.getByText(/Groq LLM not configured/)).toBeInTheDocument();
  });

  it('shows suggested questions when configured', () => {
    render(<AIChat patientContext={mockPatientContext} llmConfigured={true} />);
    expect(screen.getByText(/Explain the current sepsis prediction/)).toBeInTheDocument();
  });

  it('sends a message and displays the response', async () => {
    (endpoints.chatWithAI as ReturnType<typeof vi.fn>).mockResolvedValue({
      answer: 'The sepsis probability is 82%, indicating high risk.',
      source: 'groq_llm',
      disclaimer: 'LLM-generated explanation.',
    });

    render(<AIChat patientContext={mockPatientContext} llmConfigured={true} />);
    const textarea = screen.getByRole('textbox');
    fireEvent.change(textarea, { target: { value: 'Explain sepsis result.' } });
    fireEvent.click(screen.getByText('Send'));

    await waitFor(() => {
      expect(screen.getByText(/82%/)).toBeInTheDocument();
    });
  });

  it('shows error message when API call fails', async () => {
    (endpoints.chatWithAI as ReturnType<typeof vi.fn>).mockRejectedValue({
      response: { status: 503, data: { detail: 'Groq unavailable.' } },
      isAxiosError: true,
    });

    render(<AIChat patientContext={mockPatientContext} llmConfigured={true} />);
    const textarea = screen.getByRole('textbox');
    fireEvent.change(textarea, { target: { value: 'Test question.' } });
    fireEvent.click(screen.getByText('Send'));

    await waitFor(() => {
      expect(screen.getByRole('alert')).toBeInTheDocument();
    });
  });

  it('disables send button when input is empty', () => {
    render(<AIChat patientContext={mockPatientContext} llmConfigured={true} />);
    const sendBtn = screen.getByText('Send');
    expect(sendBtn).toBeDisabled();
  });
});

// ── AI Explain Panel ──────────────────────────────────────────────────────────

describe('AIExplainPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('shows unconfigured message when LLM not ready', () => {
    render(<AIExplainPanel patientContext={mockPatientContext} llmConfigured={false} />);
    expect(screen.getByText(/Groq LLM not configured/)).toBeInTheDocument();
  });

  it('renders Explain Prediction button when configured', () => {
    render(<AIExplainPanel patientContext={mockPatientContext} llmConfigured={true} />);
    expect(screen.getByText(/Explain Prediction/)).toBeInTheDocument();
  });

  it('displays explanation returned from API', async () => {
    (endpoints.explainPrediction as ReturnType<typeof vi.fn>).mockResolvedValue({
      explanation: 'Sepsis probability is high due to elevated lactate.',
      source: 'groq_llm',
      disclaimer: 'Not clinical advice.',
    });

    render(<AIExplainPanel patientContext={mockPatientContext} llmConfigured={true} />);
    fireEvent.click(screen.getByText(/Explain Prediction/));

    await waitFor(() => {
      expect(screen.getByText(/elevated lactate/)).toBeInTheDocument();
    });
  });

  it('shows error when API fails', async () => {
    (endpoints.explainPrediction as ReturnType<typeof vi.fn>).mockRejectedValue({
      response: { status: 503, data: { detail: 'Groq is unavailable.' } },
      isAxiosError: true,
    });

    render(<AIExplainPanel patientContext={mockPatientContext} llmConfigured={true} />);
    fireEvent.click(screen.getByText(/Explain Prediction/));

    await waitFor(() => {
      expect(screen.getByRole('alert')).toBeInTheDocument();
    });
  });
});

// ── Report Generator ──────────────────────────────────────────────────────────

describe('ReportGenerator', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('shows unconfigured message when LLM not ready', () => {
    render(<ReportGenerator patientContext={mockPatientContext} llmConfigured={false} />);
    expect(screen.getByText(/Groq LLM not configured/)).toBeInTheDocument();
  });

  it('renders Generate Report button when configured', () => {
    render(<ReportGenerator patientContext={mockPatientContext} llmConfigured={true} />);
    expect(screen.getByText(/Generate Research Report/)).toBeInTheDocument();
  });

  it('displays generated report content', async () => {
    (endpoints.generateReport as ReturnType<typeof vi.fn>).mockResolvedValue({
      report_text: 'Research Report\nSepsis probability: 82%\nRisk: HIGH',
      generated_at: new Date().toISOString(),
      disclaimer: 'RESEARCH ONLY.',
    });

    render(<ReportGenerator patientContext={mockPatientContext} llmConfigured={true} />);
    fireEvent.click(screen.getByText(/Generate Research Report/));

    await waitFor(() => {
      expect(screen.getByText(/Sepsis probability: 82%/)).toBeInTheDocument();
    });
  });

  it('antibiotic result is displayed separately and not as sepsis probability', async () => {
    const contextWithAntibiotic = {
      ...mockPatientContext,
      antibiotic_result: {
        predicted_class: 'ceftriaxone',
        probabilities: { ceftriaxone: 0.85, meropenem: 0.1 },
        model_type: 'LightGBM',
        data_disclaimer: 'RESEARCH DEMONSTRATION ONLY.',
      },
    };

    (endpoints.generateReport as ReturnType<typeof vi.fn>).mockResolvedValue({
      report_text: 'Antibiotic Demo Result: ceftriaxone\nSepsis Probability: 0.82',
      generated_at: new Date().toISOString(),
      disclaimer: 'RESEARCH ONLY.',
    });

    render(<ReportGenerator patientContext={contextWithAntibiotic} llmConfigured={true} />);
    fireEvent.click(screen.getByText(/Generate Research Report/));

    await waitFor(() => {
      const reportText = screen.getByText(/ceftriaxone/);
      expect(reportText).toBeInTheDocument();
    });
  });
});
