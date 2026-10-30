/**
 * Antibiotic Suitability Panel.
 * Calls POST /api/ai/antibiotic-suitability using the dedicated antibiotic model.
 * COMPLETELY SEPARATE from the sepsis prediction flow.
 * Shows synthetic data disclaimer prominently.
 */
import React, { useState } from 'react';
import type { AntibioticPredictionResponse } from '../../types';
import { predictAntibiotic } from '../../api/endpoints';
import { extractErrorMessage } from '../../api/client';
import { Spinner, ErrorBox, Disclaimer } from '../shared';

interface AntibioticPanelProps {
  onResultChange: (result: AntibioticPredictionResponse | null) => void;
}

const INFECTION_SOURCES = ['urinary', 'pneumonia', 'bloodstream_unknown', 'intra_abdominal', 'skin_soft_tissue'];

const DEFAULT_FEATURES = {
  age: 55,
  gender: 'M',
  heart_rate: 100,
  systolic_bp: 120,
  diastolic_bp: 80,
  map: 93,
  respiratory_rate: 18,
  temperature: 37.5,
  spo2: 98,
  wbc: 9.0,
  lactate: 1.5,
  creatinine: 1.0,
  bilirubin: 0.8,
  platelets: 200,
  glucose: 95,
  crp: 5.0,
  urine_output: 50,
  gcs: 15,
  vasopressor: 0,
  mechanical_ventilation: 0,
  infection_source_synthetic: 'urinary',
  beta_lactam_allergy_synthetic: 0,
  egfr_computed: 90,
  renal_dose_adjustment_review: 0,
};

export function AntibioticPanel({ onResultChange }: AntibioticPanelProps) {
  const [features, setFeatures] = useState<Record<string, string | number>>(DEFAULT_FEATURES);
  const [result, setResult] = useState<AntibioticPredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (key: string, value: string | number) => {
    setFeatures(prev => ({ ...prev, [key]: value }));
  };

  const handlePredict = async () => {
    if (loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    onResultChange(null);

    try {
      const payload: Record<string, unknown> = {};
      Object.entries(features).forEach(([k, v]) => {
        const num = Number(v);
        if (k === 'gender' || k === 'infection_source_synthetic') {
          payload[k] = v;
        } else {
          payload[k] = v === '' ? null : isNaN(num) ? v : num;
        }
      });

      const resp = await predictAntibiotic(payload);
      setResult(resp);
      onResultChange(resp);
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  // Sort probabilities descending
  const sortedProbs = result
    ? Object.entries(result.probabilities).sort(([, a], [, b]) => b - a)
    : [];

  return (
    <div>
      <div className="disclaimer" style={{ borderColor: 'var(--color-danger)', color: 'var(--color-danger)', background: 'rgba(239,68,68,0.08)' }}>
        ⚠️ <strong>RESEARCH DEMONSTRATION ONLY.</strong> Labels are rule-based synthetic data
        (not clinical ground truth). This model must <strong>NOT</strong> be used for clinical decisions.
        Antibiotic suitability is a <strong>separate model</strong> from sepsis prediction.
      </div>

      <div className="card">
        <h3 className="card-title">💊 Antibiotic Suitability Inputs</h3>

        {/* Categorical fields */}
        <div className="form-grid" style={{ marginBottom: '1rem' }}>
          <div className="form-group">
            <label className="form-label">Gender</label>
            <select className="form-select" value={features.gender} onChange={e => handleChange('gender', e.target.value)}>
              <option value="M">Male</option>
              <option value="F">Female</option>
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Infection Source (synthetic)</label>
            <select className="form-select" value={features.infection_source_synthetic} onChange={e => handleChange('infection_source_synthetic', e.target.value)}>
              {INFECTION_SOURCES.map(s => <option key={s} value={s}>{s.replace(/_/g, ' ')}</option>)}
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Vasopressor (0/1)</label>
            <select className="form-select" value={features.vasopressor} onChange={e => handleChange('vasopressor', Number(e.target.value))}>
              <option value={0}>No</option>
              <option value={1}>Yes</option>
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Mechanical Ventilation (0/1)</label>
            <select className="form-select" value={features.mechanical_ventilation} onChange={e => handleChange('mechanical_ventilation', Number(e.target.value))}>
              <option value={0}>No</option>
              <option value={1}>Yes</option>
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Beta-Lactam Allergy (0/1)</label>
            <select className="form-select" value={features.beta_lactam_allergy_synthetic} onChange={e => handleChange('beta_lactam_allergy_synthetic', Number(e.target.value))}>
              <option value={0}>No</option>
              <option value={1}>Yes</option>
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Renal Dose Adjustment (0/1)</label>
            <select className="form-select" value={features.renal_dose_adjustment_review} onChange={e => handleChange('renal_dose_adjustment_review', Number(e.target.value))}>
              <option value={0}>No</option>
              <option value={1}>Yes</option>
            </select>
          </div>
        </div>

        {/* Numeric fields */}
        <div className="form-grid">
          {['age','heart_rate','systolic_bp','diastolic_bp','map','respiratory_rate',
            'temperature','spo2','wbc','lactate','creatinine','bilirubin','platelets',
            'glucose','crp','urine_output','gcs','egfr_computed'].map(key => (
            <div className="form-group" key={key}>
              <label className="form-label">{key.replace(/_/g, ' ')}</label>
              <input
                type="number"
                className="form-input"
                value={features[key] ?? ''}
                onChange={e => handleChange(key, e.target.value)}
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
          {loading ? <><Spinner size={16} /> Predicting…</> : '💊 Predict Antibiotic (Demo)'}
        </button>
      </div>

      {error && <ErrorBox message={error} onRetry={handlePredict} />}

      {result && (
        <div className="card">
          <h3 className="card-title">💊 Antibiotic Demo Result</h3>

          <div style={{ marginBottom: '1rem' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--color-text-muted)' }}>Predicted class:</span>
            <div style={{
              fontSize: '1.2rem', fontWeight: 700, color: 'var(--color-primary)',
              marginTop: '0.25rem', padding: '0.5rem', background: 'var(--color-surface-2)',
              borderRadius: '8px', display: 'inline-block',
            }}>
              {result.predicted_class}
            </div>
          </div>

          <p style={{ fontSize: '0.85rem', color: 'var(--color-text-muted)', marginBottom: '0.75rem' }}>
            Class probabilities (all 10 options):
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
            {sortedProbs.map(([cls, prob]) => (
              <div key={cls} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <span style={{ width: '260px', fontSize: '0.83rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {cls === result.predicted_class ? `★ ${cls}` : cls}
                </span>
                <div style={{ flex: 1, height: '8px', background: 'var(--color-surface-2)', borderRadius: '4px', overflow: 'hidden' }}>
                  <div style={{
                    width: `${Math.round(prob * 100)}%`, height: '100%',
                    background: cls === result.predicted_class ? 'var(--color-primary)' : 'var(--color-border)',
                    borderRadius: '4px',
                  }} />
                </div>
                <span style={{ fontSize: '0.82rem', color: 'var(--color-text-muted)', width: '48px', textAlign: 'right' }}>
                  {Math.round(prob * 100)}%
                </span>
              </div>
            ))}
          </div>

          <div className="disclaimer" style={{ marginTop: '1rem' }}>
            {result.data_disclaimer}
          </div>
        </div>
      )}
    </div>
  );
}
