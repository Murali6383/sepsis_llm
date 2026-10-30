/**
 * Sepsis Prediction Dashboard Component.
 * Handles all sepsis + organ-risk prediction calls.
 * SHAP values are shown when available.
 */
import React, { useState } from 'react';
import type { SepsisPredictionResponse, OrganRiskResponse } from '../../types';
import {
  predictSepsis, predictAKI, predictCardio,
  predictLiver, predictNeuro, predictResp,
} from '../../api/endpoints';
import { extractErrorMessage } from '../../api/client';
import { Spinner, ErrorBox, RiskBadge, RiskBar, Disclaimer } from '../shared';
import { ShapDisplay } from '../ai/ShapDisplay';

interface SepsisDashboardProps {
  onResultsChange: (sepsis: SepsisPredictionResponse | null, organs: OrganRiskResponse[]) => void;
}

// Default feature values for a demo patient
const DEFAULT_SEPSIS_FEATURES = {
  age: 65,
  heart_rate: 105,
  temperature: 38.4,
  respiratory_rate: 22,
  spo2: 94,
  systolic_bp: 88,
  diastolic_bp: 58,
  map: 68,
  wbc: 14.5,
  lactate: '',
  creatinine: 1.6,
  platelets: 110,
};

export function SepsisDashboard({ onResultsChange }: SepsisDashboardProps) {
  const [features, setFeatures] = useState<Record<string, string | number>>(DEFAULT_SEPSIS_FEATURES);
  const [sepsisResult, setSepsisResult] = useState<SepsisPredictionResponse | null>(null);
  const [organResults, setOrganResults] = useState<OrganRiskResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showShap, setShowShap] = useState(false);

  const handleChange = (key: string, value: string) => {
    setFeatures(prev => ({ ...prev, [key]: value }));
  };

  const buildFeatures = () => {
    const f: Record<string, unknown> = {};
    Object.entries(features).forEach(([k, v]) => {
      const num = Number(v);
      f[k] = v === '' ? null : isNaN(num) ? v : num;
    });
    return f;
  };

  const handlePredict = async () => {
    if (loading) return;
    setLoading(true);
    setError(null);
    setSepsisResult(null);
    setOrganResults([]);

    try {
      const f = buildFeatures();
      const [sepsis, aki, cardio, liver, neuro, resp] = await Promise.allSettled([
        predictSepsis(f),
        predictAKI({ age: f.age, creatinine: f.creatinine }),
        predictCardio({ age: f.age, heart_rate: f.heart_rate, systolic_bp: f.systolic_bp, diastolic_bp: f.diastolic_bp, map: f.map }),
        predictLiver({ age: f.age }),
        predictNeuro({ age: f.age, total_gcs: null }),
        predictResp({ age: f.age, respiratory_rate: f.respiratory_rate, spo2: f.spo2 }),
      ]);

      const sepsisData = sepsis.status === 'fulfilled' ? sepsis.value : null;
      const organData = [aki, cardio, liver, neuro, resp]
        .filter(r => r.status === 'fulfilled')
        .map(r => (r as PromiseFulfilledResult<OrganRiskResponse>).value);

      setSepsisResult(sepsisData);
      setOrganResults(organData);
      onResultsChange(sepsisData, organData);

      if (sepsis.status === 'rejected') {
        setError(`Sepsis prediction failed: ${extractErrorMessage(sepsis.reason)}`);
      }
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const organLabel: Record<string, string> = {
    acute_kidney_injury: '🫘 AKI',
    cardiovascular: '❤️ Cardiac',
    liver: '🍀 Liver',
    neurological: '🧠 Neuro',
    respiratory: '🫁 Respiratory',
  };

  return (
    <div>
      <Disclaimer />

      {/* Feature input form */}
      <div className="card">
        <h3 className="card-title">Clinical Features</h3>
        <p style={{ fontSize: '0.82rem', color: 'var(--color-text-muted)', marginBottom: '1rem' }}>
          Enter available values. Leave blank for missing data (imputed by model).
        </p>
        <div className="form-grid">
          {Object.entries(DEFAULT_SEPSIS_FEATURES).map(([key]) => (
            <div className="form-group" key={key}>
              <label className="form-label">{key.replace(/_/g, ' ')}</label>
              <input
                type="number"
                className="form-input"
                value={features[key] ?? ''}
                onChange={e => handleChange(key, e.target.value)}
                placeholder="Optional"
              />
            </div>
          ))}
        </div>

        <button
          className="btn btn-primary"
          onClick={handlePredict}
          disabled={loading}
          style={{ marginTop: '0.5rem' }}
        >
          {loading ? <><Spinner size={16} /> Running Predictions…</> : '🔍 Run All Predictions'}
        </button>
      </div>

      {error && <ErrorBox message={error} onRetry={handlePredict} />}

      {/* Sepsis result */}
      {sepsisResult && (
        <div className="card">
          <h3 className="card-title">🦠 Sepsis Risk Prediction</h3>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '0.75rem', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '2rem', fontWeight: 700, color: sepsisResult.risk_level === 'high' ? 'var(--color-danger)' : sepsisResult.risk_level === 'moderate' ? 'var(--color-warning)' : 'var(--color-success)' }}>
              {Math.round(sepsisResult.sepsis_probability * 100)}%
            </span>
            <RiskBadge level={sepsisResult.risk_level} />
            <span style={{ color: 'var(--color-text-muted)', fontSize: '0.85rem' }}>
              Model: {sepsisResult.model_type} | Threshold: {sepsisResult.threshold}
            </span>
          </div>
          <RiskBar probability={sepsisResult.sepsis_probability} level={sepsisResult.risk_level} />
          <p style={{ fontSize: '0.8rem', color: 'var(--color-text-muted)', marginTop: '0.5rem' }}>
            Label: {sepsisResult.sepsis_label === 1 ? 'Positive (≥ threshold)' : 'Negative (< threshold)'}
          </p>

          {sepsisResult.shap_available && sepsisResult.shap_values && (
            <div style={{ marginTop: '1rem' }}>
              <button
                className="btn btn-outline btn-sm"
                onClick={() => setShowShap(v => !v)}
              >
                {showShap ? '▲ Hide SHAP' : '▼ Show SHAP Explanations'}
              </button>
              {showShap && (
                <div style={{ marginTop: '0.75rem' }}>
                  <ShapDisplay shapValues={sepsisResult.shap_values} />
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Organ risk results */}
      {organResults.length > 0 && (
        <div className="card">
          <h3 className="card-title">🫀 Organ Risk Predictions</h3>
          <p style={{ fontSize: '0.8rem', color: 'var(--color-text-muted)', marginBottom: '1rem' }}>
            Each organ model is independent — scores are not derived from sepsis probability.
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '0.75rem' }}>
            {organResults.map((organ, i) => (
              <div
                key={i}
                style={{
                  background: 'var(--color-surface-2)',
                  borderRadius: '8px',
                  padding: '0.85rem',
                  border: '1px solid var(--color-border)',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                  <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>
                    {organLabel[organ.organ] ?? organ.organ}
                  </span>
                  <RiskBadge level={organ.risk_level} />
                </div>
                <RiskBar probability={organ.probability} level={organ.risk_level} />
                <p style={{ fontSize: '0.8rem', color: 'var(--color-text-muted)', marginTop: '0.35rem' }}>
                  {Math.round(organ.probability * 100)}% probability
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
