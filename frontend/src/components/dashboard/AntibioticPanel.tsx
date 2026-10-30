/**
 * Antimicrobial Susceptibility Analysis Panel.
 * Uses the LightGBM classifier trained on the real-world ARMD-ECUH microbiology cohort
 * (1.93M culture records across 62,387 patients, Dryad DOI: 10.5061/dryad.7sqv9s55x).
 *
 * In-vitro phenotype prediction (Susceptible / Resistant / Intermediate)
 * with patient-level GroupShuffleSplit validation and TreeSHAP attribution.
 */
import React, { useState } from 'react';
import type { AntibioticPredictionResponse, AntibioticFeaturesRequest } from '../../types';
import { predictAntibiotic } from '../../api/endpoints';
import { extractErrorMessage } from '../../api/client';
import { Spinner, ErrorBox } from '../shared';

interface AntibioticPanelProps {
  onResultChange: (result: AntibioticPredictionResponse | null) => void;
}

const COMMON_ORGANISMS = [
  'ESCHERICHIA COLI',
  'KLEBSIELLA PNEUMONIAE',
  'PSEUDOMONAS AERUGINOSA',
  'STAPHYLOCOCCUS AUREUS',
  'ENTEROCOCCUS FAECALIS',
  'PROTEUS MIRABILIS',
  'ENTEROBACTER CLOACAE',
  'ACINETOBACTER BAUMANNII',
  'SERRATIA MARCESCENS',
  'ENTEROCOCCUS FAECIUM',
  'STAPHYLOCOCCUS EPIDERMIDIS',
  'KLEBSIELLA OXYTOCA',
];

const COMMON_ANTIBIOTICS = [
  'Ciprofloxacin',
  'Ceftriaxone',
  'Piperacillin/tazobactam',
  'Vancomycin',
  'Meropenem',
  'Ampicillin',
  'Gentamicin',
  'Levofloxacin',
  'Cefepime',
  'Trimethoprim/sulfamethoxazole',
  'Tobramycin',
  'Aztreonam',
  'Amikacin',
  'Ertapenem',
];

const CULTURE_DESCRIPTIONS = [
  'URINE',
  'BLOOD',
  'RESPIRATORY',
  'WOUND',
  'BODY FLUID',
  'CATHETER TIP',
  'STOOL',
];

const ORDERING_MODES = [
  'Inpatient',
  'Outpatient',
  'Emergency',
];

export function AntibioticPanel({ onResultChange }: AntibioticPanelProps) {
  const [form, setForm] = useState<AntibioticFeaturesRequest>({
    organism: 'ESCHERICHIA COLI',
    antibiotic: 'Ciprofloxacin',
    culture_description: 'URINE',
    ordering_mode: 'Inpatient',
  });

  const [result, setResult] = useState<AntibioticPredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (field: keyof AntibioticFeaturesRequest, value: string) => {
    setForm(prev => ({ ...prev, [field]: value }));
  };

  const handlePredict = async () => {
    if (loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    onResultChange(null);

    try {
      const resp = await predictAntibiotic({
        organism: form.organism,
        antibiotic: form.antibiotic,
        culture_description: form.culture_description,
        ordering_mode: form.ordering_mode,
      });
      setResult(resp);
      onResultChange(resp);
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const getBadgeStyle = (cls: string) => {
    switch (cls) {
      case 'Susceptible':
        return { background: '#ecfdf5', color: '#047857', border: '1px solid #a7f3d0' };
      case 'Resistant':
        return { background: '#fef2f2', color: '#b91c1c', border: '1px solid #fecaca' };
      case 'Intermediate':
        return { background: '#fffbeb', color: '#b45309', border: '1px solid #fde68a' };
      default:
        return { background: '#f1f5f9', color: '#475569', border: '1px solid #cbd5e1' };
    }
  };

  return (
    <div>
      {/* Required Clinical Dataset Disclaimer */}
      <div
        className="disclaimer"
        style={{
          borderLeft: '4px solid #4f46e5',
          background: '#f8fafc',
          padding: '1rem',
          borderRadius: '8px',
          color: '#334155',
          marginBottom: '1.25rem',
          boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
        }}
      >
        <p style={{ margin: 0, fontWeight: 600, color: '#1e293b' }}>
          Data source: ARMD-ECUH retrospective research dataset.
        </p>
        <p style={{ margin: '0.35rem 0 0', fontSize: '0.85rem', lineHeight: 1.5 }}>
          Model outputs are research estimates and have not been prospectively or clinically validated.
          Antimicrobial susceptibility prediction is an in-vitro phenotype classification and does not
          constitute an individualized clinical treatment recommendation.
        </p>
      </div>

      {/* Input Selection Form */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
          <h3 className="card-title" style={{ margin: 0 }}>
            🔬 Culture & Antimicrobial Parameters
          </h3>
          <span style={{ fontSize: '0.75rem', background: '#e0e7ff', color: '#3730a3', padding: '0.2rem 0.6rem', borderRadius: '9999px', fontWeight: 600 }}>
            ARMD-ECUH Verified
          </span>
        </div>

        <div className="form-grid" style={{ marginBottom: '1.25rem' }}>
          <div className="form-group">
            <label className="form-label">Isolated Microorganism</label>
            <select
              className="form-select"
              value={form.organism}
              onChange={e => handleChange('organism', e.target.value)}
            >
              {COMMON_ORGANISMS.map(org => (
                <option key={org} value={org}>
                  {org}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Antimicrobial Agent</label>
            <select
              className="form-select"
              value={form.antibiotic}
              onChange={e => handleChange('antibiotic', e.target.value)}
            >
              {COMMON_ANTIBIOTICS.map(abx => (
                <option key={abx} value={abx}>
                  {abx}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Specimen Source / Culture Site</label>
            <select
              className="form-select"
              value={form.culture_description}
              onChange={e => handleChange('culture_description', e.target.value)}
            >
              {CULTURE_DESCRIPTIONS.map(site => (
                <option key={site} value={site}>
                  {site}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Ordering Encounter Mode</label>
            <select
              className="form-select"
              value={form.ordering_mode}
              onChange={e => handleChange('ordering_mode', e.target.value)}
            >
              {ORDERING_MODES.map(mode => (
                <option key={mode} value={mode}>
                  {mode}
                </option>
              ))}
            </select>
          </div>
        </div>

        <button
          className="btn btn-primary"
          onClick={handlePredict}
          disabled={loading}
          style={{ width: '100%', padding: '0.75rem', fontWeight: 600 }}
        >
          {loading ? (
            <>
              <Spinner size={16} /> Computing In-Vitro Susceptibility…
            </>
          ) : (
            '⚡ Evaluate Antimicrobial Susceptibility'
          )}
        </button>
      </div>

      {error && <ErrorBox message={error} onRetry={handlePredict} />}

      {/* Prediction Output Card */}
      {result && (
        <div className="card" style={{ marginBottom: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
            <h3 className="card-title" style={{ margin: 0 }}>
              📊 Predicted Phenotype Result
            </h3>
            <span
              style={{
                fontSize: '0.85rem',
                fontWeight: 700,
                padding: '0.35rem 0.85rem',
                borderRadius: '6px',
                ...getBadgeStyle(result.predicted_class),
              }}
            >
              {result.predicted_class.toUpperCase()}
            </span>
          </div>

          <div style={{ background: '#f8fafc', padding: '1rem', borderRadius: '8px', marginBottom: '1.25rem' }}>
            <div style={{ fontSize: '0.85rem', color: '#64748b', marginBottom: '0.35rem' }}>
              Target Pathogen & Drug Combination:
            </div>
            <div style={{ fontSize: '1.05rem', fontWeight: 600, color: '#1e293b' }}>
              <em>{result.organism || form.organism}</em> vs. {result.antibiotic || form.antibiotic}
            </div>
            <div style={{ fontSize: '0.8rem', color: '#64748b', marginTop: '0.25rem' }}>
              Specimen: {result.culture_description || form.culture_description} | Setting: {result.ordering_mode || form.ordering_mode}
            </div>
          </div>

          {/* Probability Distribution */}
          <div style={{ marginBottom: '1.5rem' }}>
            <h4 style={{ fontSize: '0.9rem', fontWeight: 600, color: '#334155', marginBottom: '0.75rem' }}>
              Class Probability Distribution
            </h4>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
              {['Susceptible', 'Resistant', 'Intermediate'].map(cls => {
                const prob = result.probabilities[cls] ?? 0;
                const pct = Math.round(prob * 100);
                const isWinner = result.predicted_class === cls;
                const barColor = cls === 'Susceptible' ? '#10b981' : cls === 'Resistant' ? '#ef4444' : '#f59e0b';

                return (
                  <div key={cls}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', marginBottom: '0.2rem' }}>
                      <span style={{ fontWeight: isWinner ? 700 : 500, color: isWinner ? '#0f172a' : '#64748b' }}>
                        {isWinner ? `✓ ${cls}` : cls}
                      </span>
                      <span style={{ fontWeight: 600, color: isWinner ? '#0f172a' : '#64748b' }}>
                        {(prob * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div style={{ height: '8px', background: '#e2e8f0', borderRadius: '9999px', overflow: 'hidden' }}>
                      <div
                        style={{
                          width: `${pct}%`,
                          height: '100%',
                          background: barColor,
                          borderRadius: '9999px',
                          transition: 'width 0.4s ease-in-out',
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* TreeSHAP Feature Attribution */}
          {result.shap_available && result.shap_values && result.shap_values.length > 0 && (
            <div style={{ marginBottom: '1.5rem', background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: '8px', padding: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                <h4 style={{ fontSize: '0.88rem', fontWeight: 600, color: '#334155', margin: 0 }}>
                  🔍 TreeSHAP Feature Attributions
                </h4>
                <span style={{ fontSize: '0.75rem', color: '#64748b' }}>Impact on Resistance Odds</span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {result.shap_values.map((item, idx) => {
                  const val = item.shap_value;
                  const isPositive = val >= 0;
                  const width = Math.min(Math.abs(val) * 60, 100);

                  return (
                    <div key={idx} style={{ display: 'flex', alignItems: 'center', fontSize: '0.8rem' }}>
                      <div style={{ width: '150px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontWeight: 500, color: '#475569' }}>
                        {item.feature}
                      </div>
                      <div style={{ flex: 1, display: 'flex', alignItems: 'center', height: '14px', background: '#f1f5f9', borderRadius: '4px', position: 'relative' }}>
                        <div
                          style={{
                            width: `${Math.max(width, 6)}%`,
                            height: '100%',
                            background: isPositive ? '#f87171' : '#34d399',
                            borderRadius: '4px',
                          }}
                        />
                      </div>
                      <div style={{ width: '60px', textAlign: 'right', fontWeight: 600, color: isPositive ? '#dc2626' : '#059669', fontSize: '0.75rem' }}>
                        {isPositive ? `+${val.toFixed(2)}` : val.toFixed(2)}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Model Provenance and Metadata */}
          <div style={{ background: '#f8fafc', padding: '0.85rem', borderRadius: '6px', fontSize: '0.78rem', color: '#64748b', border: '1px solid #e2e8f0' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
              <span>Architecture: <strong>{result.model_type}</strong> (v{result.model_version || '1.0.0'})</span>
              <span>Dataset: <strong>ARMD-ECUH</strong> (Dryad)</span>
            </div>
            <div>Validation: Patient-level GroupShuffleSplit on <code>anon_id</code> (Zero test patient leakage)</div>
          </div>
        </div>
      )}

      {/* Model Benchmark Performance Card */}
      <div className="card" style={{ background: '#ffffff', border: '1px solid #e2e8f0' }}>
        <h4 style={{ fontSize: '0.92rem', fontWeight: 600, color: '#1e293b', marginBottom: '0.75rem' }}>
          📈 Empirical Model Benchmark Comparison (ARMD-ECUH Held-Out Patients)
        </h4>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem', textAlign: 'left' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', color: '#475569' }}>
                <th style={{ padding: '0.5rem' }}>Model Architecture</th>
                <th style={{ padding: '0.5rem' }}>Balanced Acc</th>
                <th style={{ padding: '0.5rem' }}>Macro ROC-AUC</th>
                <th style={{ padding: '0.5rem' }}>Macro F1</th>
                <th style={{ padding: '0.5rem' }}>Status</th>
              </tr>
            </thead>
            <tbody>
              <tr style={{ background: '#eff6ff', fontWeight: 600, borderBottom: '1px solid #e2e8f0' }}>
                <td style={{ padding: '0.5rem', color: '#1d4ed8' }}>LightGBM Classifier</td>
                <td style={{ padding: '0.5rem' }}>71.86%</td>
                <td style={{ padding: '0.5rem' }}>0.8727</td>
                <td style={{ padding: '0.5rem' }}>0.5492</td>
                <td style={{ padding: '0.5rem', color: '#15803d' }}>★ Deployed</td>
              </tr>
              <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '0.5rem', color: '#334155' }}>Random Forest (100 Trees)</td>
                <td style={{ padding: '0.5rem' }}>69.29%</td>
                <td style={{ padding: '0.5rem' }}>0.8535</td>
                <td style={{ padding: '0.5rem' }}>0.5389</td>
                <td style={{ padding: '0.5rem', color: '#64748b' }}>Benchmark Baseline</td>
              </tr>
              <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '0.5rem', color: '#334155' }}>Logistic Regression (Balanced)</td>
                <td style={{ padding: '0.5rem' }}>66.08%</td>
                <td style={{ padding: '0.5rem' }}>0.8270</td>
                <td style={{ padding: '0.5rem' }}>0.4874</td>
                <td style={{ padding: '0.5rem', color: '#64748b' }}>Linear Baseline</td>
              </tr>
              <tr>
                <td style={{ padding: '0.5rem', color: '#334155' }}>XGBoost Baseline</td>
                <td style={{ padding: '0.5rem' }}>49.05%</td>
                <td style={{ padding: '0.5rem' }}>0.7841</td>
                <td style={{ padding: '0.5rem' }}>0.3687</td>
                <td style={{ padding: '0.5rem', color: '#64748b' }}>Imbalanced Baseline</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p style={{ margin: '0.6rem 0 0', fontSize: '0.75rem', color: '#94a3b8' }}>
          * Evaluated on held-out test encounters with patient grouping on anon_id. LightGBM was selected for deployment due to highest Balanced Accuracy and Macro ROC-AUC.
        </p>
      </div>
    </div>
  );
}
