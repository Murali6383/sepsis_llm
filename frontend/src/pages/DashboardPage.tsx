/**
 * Main Dashboard Page.
 * Orchestrates all prediction panels and AI features.
 * Maintains shared patient context for LLM features.
 */
import React, { useState, useEffect } from 'react';
import type { SepsisPredictionResponse, OrganRiskResponse, AntibioticPredictionResponse, AIHealthResponse } from '../types';
import { getAIHealth } from '../api/endpoints';
import { SepsisDashboard } from '../components/dashboard/SepsisDashboard';
import { AntibioticPanel } from '../components/dashboard/AntibioticPanel';
import { AIExplainPanel } from '../components/ai/AIExplainPanel';
import { AIChat } from '../components/ai/AIChat';
import { ReportGenerator } from '../components/ai/ReportGenerator';
import { Spinner } from '../components/shared';

type Tab = 'sepsis' | 'antibiotic' | 'explain' | 'chat' | 'report';

interface DashboardPageProps {
  username: string | null;
  role: string | null;
  onLogout: () => void;
}

export function DashboardPage({ username, role, onLogout }: DashboardPageProps) {
  const [activeTab, setActiveTab] = useState<Tab>('sepsis');
  const [aiHealth, setAIHealth] = useState<AIHealthResponse | null>(null);
  const [aiHealthLoading, setAIHealthLoading] = useState(true);

  // Shared patient context — filled by prediction components
  const [sepsisResult, setSepsisResult] = useState<SepsisPredictionResponse | null>(null);
  const [organResults, setOrganResults] = useState<OrganRiskResponse[]>([]);
  const [antibioticResult, setAntibioticResult] = useState<AntibioticPredictionResponse | null>(null);

  useEffect(() => {
    getAIHealth()
      .then(setAIHealth)
      .catch(() => setAIHealth(null))
      .finally(() => setAIHealthLoading(false));
  }, []);

  const patientContext = {
    sepsis_result: sepsisResult,
    organ_results: organResults,
    antibiotic_result: antibioticResult,
  };

  const llmConfigured = aiHealth?.llm_configured ?? false;

  const tabs: { id: Tab; label: string }[] = [
    { id: 'sepsis', label: '🦠 Sepsis & Organs' },
    { id: 'antibiotic', label: '💊 Antibiotic Demo' },
    { id: 'explain', label: '✨ AI Explain' },
    { id: 'chat', label: '🤖 AI Chat' },
    { id: 'report', label: '📄 Research Report' },
  ];

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <nav className="sidebar">
        <div className="sidebar-brand">
          <span>🛡️</span>
          <span>SepsisGuardian</span>
        </div>

        {tabs.map(t => (
          <button
            key={t.id}
            className={`sidebar-link ${activeTab === t.id ? 'active' : ''}`}
            onClick={() => setActiveTab(t.id)}
            style={{ background: 'none', border: 'none', textAlign: 'left', cursor: 'pointer', width: '100%' }}
          >
            <span>{t.label}</span>
          </button>
        ))}

        <div style={{ marginTop: 'auto', paddingTop: '1rem', borderTop: '1px solid var(--color-border)' }}>
          {/* AI Health indicator */}
          {aiHealthLoading ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem', fontSize: '0.8rem', color: 'var(--color-text-muted)' }}>
              <Spinner size={12} /> LLM status…
            </div>
          ) : (
            <div style={{ padding: '0.5rem', fontSize: '0.78rem', color: 'var(--color-text-muted)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <span className={`status-dot ${llmConfigured ? 'ok' : 'warn'}`} />
                LLM: {llmConfigured ? aiHealth?.llm_model ?? 'ready' : 'not configured'}
              </div>
            </div>
          )}

          <div style={{ padding: '0.5rem', fontSize: '0.78rem', color: 'var(--color-text-muted)' }}>
            <div>{username}</div>
            <div style={{ opacity: 0.7 }}>{role}</div>
          </div>

          <button className="btn btn-outline btn-sm" onClick={onLogout} style={{ width: '100%', marginTop: '0.25rem' }}>
            Log Out
          </button>
        </div>
      </nav>

      {/* Main content */}
      <main className="main-content">
        {activeTab === 'sepsis' && (
          <div>
            <h1 style={{ fontSize: '1.3rem', fontWeight: 700, marginBottom: '1.5rem' }}>
              Sepsis Risk & Organ Dysfunction Predictions
            </h1>
            <SepsisDashboard
              onResultsChange={(s, o) => { setSepsisResult(s); setOrganResults(o); }}
            />
          </div>
        )}

        {activeTab === 'antibiotic' && (
          <div>
            <h1 style={{ fontSize: '1.3rem', fontWeight: 700, marginBottom: '1.5rem' }}>
              Antibiotic Suitability Demo
            </h1>
            <p style={{ color: 'var(--color-text-muted)', fontSize: '0.875rem', marginBottom: '1.5rem' }}>
              This panel uses a <strong>separate LightGBM multiclass model</strong> with its own inputs.
              It is entirely independent of the sepsis prediction model.
            </p>
            <AntibioticPanel onResultChange={setAntibioticResult} />
          </div>
        )}

        {activeTab === 'explain' && (
          <div>
            <h1 style={{ fontSize: '1.3rem', fontWeight: 700, marginBottom: '0.5rem' }}>
              AI Prediction Explanation
            </h1>
            <p style={{ color: 'var(--color-text-muted)', fontSize: '0.875rem', marginBottom: '1.5rem' }}>
              The LLM explains actual model outputs — it does not generate replacement scores.
              Run predictions first.
            </p>
            <div className="card">
              <AIExplainPanel patientContext={patientContext} llmConfigured={llmConfigured} />
            </div>
          </div>
        )}

        {activeTab === 'chat' && (
          <div>
            <h1 style={{ fontSize: '1.3rem', fontWeight: 700, marginBottom: '0.5rem' }}>
              AI Research Assistant
            </h1>
            <p style={{ color: 'var(--color-text-muted)', fontSize: '0.875rem', marginBottom: '1.5rem' }}>
              Ask questions about the current prediction results. Powered by Groq LLM.
            </p>
            <div className="card">
              <AIChat patientContext={patientContext} llmConfigured={llmConfigured} />
            </div>
          </div>
        )}

        {activeTab === 'report' && (
          <div>
            <h1 style={{ fontSize: '1.3rem', fontWeight: 700, marginBottom: '0.5rem' }}>
              Research Report Generator
            </h1>
            <p style={{ color: 'var(--color-text-muted)', fontSize: '0.875rem', marginBottom: '1.5rem' }}>
              Generates a structured report from actual model outputs. No fabricated values.
            </p>
            <div className="card">
              <ReportGenerator patientContext={patientContext} llmConfigured={llmConfigured} />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
