/**
 * Patient-Specific Antibiotic Suitability Prediction & Antimicrobial Susceptibility Analysis.
 *
 * Model architecture: LightGBM + Patient-specific feature analysis + TreeSHAP explainability.
 * Integrated with the real-world ARMD-ECUH retrospective research dataset
 * (Dryad DOI: 10.5061/dryad.7sqv9s55x, N=1,931,685 culture tests across 62,387 patients).
 */
import React, { useState } from 'react';
import type {
  CandidateAntibioticRanking,
  PatientAntibioticSuitabilityResponse,
  AntibioticPredictionResponse,
} from '../../types';
import {
  predictPatientAntibioticSuitability,
  predictAntibiotic,
} from '../../api/endpoints';
import { extractErrorMessage } from '../../api/client';
import { Spinner, ErrorBox } from '../shared';

interface AntibioticPanelProps {
  onResultChange: (
    result:
      | PatientAntibioticSuitabilityResponse
      | AntibioticPredictionResponse
      | null
  ) => void;
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

const INFECTION_SITES = [
  'URINE',
  'BLOOD',
  'RESPIRATORY',
  'WOUND',
  'ABDOMINAL',
  'BODY FLUID',
  'CATHETER TIP',
];

const ALLERGY_OPTIONS = [
  'Penicillins',
  'Cephalosporins',
  'Fluoroquinolones',
  'Sulfonamides',
  'Glycopeptides (Vancomycin)',
  'Aminoglycosides',
];

const RESISTANCE_PROFILE_OPTIONS = [
  'ESBL (Extended-Spectrum Beta-Lactamase)',
  'MRSA (Methicillin-Resistant S. aureus)',
  'VRE (Vancomycin-Resistant Enterococci)',
  'CRE (Carbapenem-Resistant Enterobacterales)',
  'Fluoroquinolone-Resistant',
];

const MEDICATION_OPTIONS = [
  'Warfarin (Anticoagulant)',
  'NSAIDs (Ibuprofen/Ketorolac)',
  'Amiodarone (QT prolonging agent)',
  'ACE Inhibitors / ARBs',
  'Methotrexate',
];

const PREVIOUS_EXPOSURE_OPTIONS = [
  'Ciprofloxacin',
  'Ceftriaxone',
  'Piperacillin/tazobactam',
  'Meropenem',
  'Vancomycin',
  'Levofloxacin',
  'Gentamicin',
];

const COMMON_ANTIBIOTICS = [
  'Ceftriaxone',
  'Ciprofloxacin',
  'Piperacillin/tazobactam',
  'Meropenem',
  'Vancomycin',
  'Levofloxacin',
  'Cefepime',
  'Ampicillin',
  'Gentamicin',
  'Trimethoprim/sulfamethoxazole',
  'Aztreonam',
  'Tobramycin',
];

export function AntibioticPanel({ onResultChange }: AntibioticPanelProps) {
  const [activeMode, setActiveMode] = useState<'patient_specific' | 'in_vitro_direct'>('patient_specific');

  // ── Patient-Specific Form State ──
  const [patientAge, setPatientAge] = useState<number>(65);
  const [patientGender, setPatientGender] = useState<string>('M');
  const [infectionSite, setInfectionSite] = useState<string>('URINE');
  const [identifiedOrganism, setIdentifiedOrganism] = useState<string>('ESCHERICHIA COLI');

  const [drugAllergies, setDrugAllergies] = useState<string[]>([]);
  const [resistanceProfile, setResistanceProfile] = useState<string[]>([]);
  const [currentMedications, setCurrentMedications] = useState<string[]>([]);
  const [previousExposure, setPreviousExposure] = useState<string[]>([]);

  const [creatinine, setCreatinine] = useState<number>(1.2);
  const [egfr, setEgfr] = useState<number>(65);
  const [totalBilirubin, setTotalBilirubin] = useState<string>('1.0');
  const [astAltElevated, setAstAltElevated] = useState<boolean>(false);

  // ── In-Vitro Direct Form State ──
  const [directAntibiotic, setDirectAntibiotic] = useState<string>('Ciprofloxacin');

  // ── Output States ──
  const [suitabilityResponse, setSuitabilityResponse] = useState<PatientAntibioticSuitabilityResponse | null>(null);
  const [directResponse, setDirectResponse] = useState<AntibioticPredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedCandidate, setExpandedCandidate] = useState<string | null>(null);

  const toggleArrayItem = (item: string, currentList: string[], setter: (v: string[]) => void) => {
    if (currentList.includes(item)) {
      setter(currentList.filter(x => x !== item));
    } else {
      setter([...currentList, item]);
    }
  };

  const handleRunPatientSuitability = async () => {
    if (loading) return;
    setLoading(true);
    setError(null);
    setSuitabilityResponse(null);
    setDirectResponse(null);
    onResultChange(null);

    try {
      const resp = await predictPatientAntibioticSuitability({
        patient_age: Number(patientAge),
        patient_gender: patientGender,
        infection_site: infectionSite,
        identified_organism: identifiedOrganism,
        drug_allergies: drugAllergies,
        antibiotic_resistance_profile: resistanceProfile,
        current_medications: currentMedications,
        previous_antibiotic_exposure: previousExposure,
        creatinine: Number(creatinine),
        egfr: Number(egfr),
        total_bilirubin: totalBilirubin ? Number(totalBilirubin) : null,
        ast_alt_elevated: astAltElevated,
      });

      // Augment response with aliases for report/explain panel compatibility
      const topCand = resp.ranked_candidates[0];
      const augmented = {
        ...resp,
        predicted_class: topCand ? `${topCand.antibiotic} (${topCand.predicted_susceptibility}, Score: ${topCand.suitability_score}%)` : 'Unknown',
        data_disclaimer: resp.disclaimer,
      };

      setSuitabilityResponse(resp);
      onResultChange(augmented);
      if (resp.ranked_candidates.length > 0) {
        setExpandedCandidate(resp.ranked_candidates[0].antibiotic);
      }
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const handleRunDirectInVitro = async () => {
    if (loading) return;
    setLoading(true);
    setError(null);
    setSuitabilityResponse(null);
    setDirectResponse(null);
    onResultChange(null);

    try {
      const resp = await predictAntibiotic({
        organism: identifiedOrganism,
        antibiotic: directAntibiotic,
        culture_description: infectionSite,
        ordering_mode: 'Inpatient',
      });
      setDirectResponse(resp);
      onResultChange(resp);
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 75) return '#10b981'; // Green
    if (score >= 45) return '#f59e0b'; // Amber
    return '#ef4444'; // Red
  };

  const getSusceptibilityBadge = (cls: string) => {
    switch (cls) {
      case 'Susceptible':
        return { bg: '#ecfdf5', color: '#047857', border: '#a7f3d0' };
      case 'Intermediate':
        return { bg: '#fffbeb', color: '#b45309', border: '#fde68a' };
      case 'Resistant':
        return { bg: '#fef2f2', color: '#b91c1c', border: '#fecaca' };
      default:
        return { bg: '#f1f5f9', color: '#475569', border: '#cbd5e1' };
    }
  };

  return (
    <div>
      {/* Required Dataset & Clinical Governance Disclaimer */}
      <div
        className="disclaimer"
        style={{
          borderLeft: '4px solid #4f46e5',
          background: '#ffffff',
          padding: '1rem',
          borderRadius: '8px',
          color: '#334155',
          marginBottom: '1.25rem',
          boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
          border: '1px solid #e2e8f0',
          borderLeftWidth: '4px',
        }}
      >
        <p style={{ margin: 0, fontWeight: 700, color: '#1e293b', fontSize: '0.92rem' }}>
          Data source: ARMD-ECUH retrospective research dataset (Dryad DOI: 10.5061/dryad.7sqv9s55x).
        </p>
        <p style={{ margin: '0.35rem 0 0', fontSize: '0.85rem', lineHeight: 1.5, color: '#475569' }}>
          Predictions are research estimates and require clinical validation. They are not prescriptions or
          substitutes for antimicrobial stewardship and clinician judgment.
        </p>
      </div>

      {/* Mode Switcher Tabs */}
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.25rem' }}>
        <button
          className={`btn ${activeMode === 'patient_specific' ? 'btn-primary' : 'btn-outline'}`}
          onClick={() => setActiveMode('patient_specific')}
          style={{ fontSize: '0.88rem', fontWeight: 600, padding: '0.5rem 1rem' }}
        >
          🎯 Patient-Specific Suitability Ranking
        </button>
        <button
          className={`btn ${activeMode === 'in_vitro_direct' ? 'btn-primary' : 'btn-outline'}`}
          onClick={() => setActiveMode('in_vitro_direct')}
          style={{ fontSize: '0.88rem', fontWeight: 600, padding: '0.5rem 1rem' }}
        >
          🔬 Direct In-Vitro AST Phenotype
        </button>
      </div>

      {/* Mode 1: Patient-Specific Antibiotic Suitability */}
      {activeMode === 'patient_specific' && (
        <div className="card" style={{ marginBottom: '1.5rem', background: '#ffffff', border: '1px solid #e2e8f0' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.5rem' }}>
            <div>
              <h3 className="card-title" style={{ margin: 0, color: '#0f172a' }}>
                💊 Patient Clinical Parameters & Antimicrobial Context
              </h3>
              <p style={{ margin: '0.25rem 0 0', fontSize: '0.82rem', color: '#64748b' }}>
                Combines ARMD-ECUH in-vitro machine learning with organ function, allergy, and drug interaction safety checks.
              </p>
            </div>
            <span style={{ fontSize: '0.75rem', background: '#e0e7ff', color: '#3730a3', padding: '0.25rem 0.65rem', borderRadius: '9999px', fontWeight: 700 }}>
              LightGBM + Clinical Safety Engine
            </span>
          </div>

          {/* Core Demographics & Infection */}
          <div style={{ background: '#f8fafc', padding: '1rem', borderRadius: '8px', marginBottom: '1.25rem', border: '1px solid #edf2f7' }}>
            <h4 style={{ fontSize: '0.85rem', fontWeight: 700, color: '#334155', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.75rem' }}>
              1. Infection & Demographics
            </h4>
            <div className="form-grid">
              <div className="form-group">
                <label className="form-label">Identified Microorganism</label>
                <select
                  className="form-select"
                  value={identifiedOrganism}
                  onChange={e => setIdentifiedOrganism(e.target.value)}
                >
                  {COMMON_ORGANISMS.map(org => (
                    <option key={org} value={org}>
                      {org}
                    </option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Infection Site / Specimen</label>
                <select
                  className="form-select"
                  value={infectionSite}
                  onChange={e => setInfectionSite(e.target.value)}
                >
                  {INFECTION_SITES.map(site => (
                    <option key={site} value={site}>
                      {site}
                    </option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Patient Age</label>
                <input
                  type="number"
                  className="form-input"
                  value={patientAge}
                  onChange={e => setPatientAge(Number(e.target.value))}
                  min={0}
                  max={120}
                />
              </div>

              <div className="form-group">
                <label className="form-label">Gender</label>
                <select
                  className="form-select"
                  value={patientGender}
                  onChange={e => setPatientGender(e.target.value)}
                >
                  <option value="M">Male</option>
                  <option value="F">Female</option>
                  <option value="Other">Other</option>
                </select>
              </div>
            </div>
          </div>

          {/* Organ Functions (Kidney & Liver) */}
          <div style={{ background: '#f8fafc', padding: '1rem', borderRadius: '8px', marginBottom: '1.25rem', border: '1px solid #edf2f7' }}>
            <h4 style={{ fontSize: '0.85rem', fontWeight: 700, color: '#334155', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.75rem' }}>
              2. Renal & Hepatic Function (Dosing & Toxicity)
            </h4>
            <div className="form-grid">
              <div className="form-group">
                <label className="form-label">Serum Creatinine (mg/dL)</label>
                <input
                  type="number"
                  step="0.1"
                  className="form-input"
                  value={creatinine}
                  onChange={e => setCreatinine(Number(e.target.value))}
                  min={0.1}
                  max={25}
                />
              </div>

              <div className="form-group">
                <label className="form-label">eGFR (mL/min/1.73m²)</label>
                <input
                  type="number"
                  className="form-input"
                  value={egfr}
                  onChange={e => setEgfr(Number(e.target.value))}
                  min={1}
                  max={200}
                />
                <span style={{ fontSize: '0.72rem', color: egfr < 30 ? '#dc2626' : egfr < 60 ? '#d97706' : '#16a34a', fontWeight: 600 }}>
                  {egfr < 30 ? '⚠️ Severe CKD (eGFR < 30)' : egfr < 60 ? '⚠️ Moderate CKD (eGFR 30-59)' : '✓ Normal / Mild'}
                </span>
              </div>

              <div className="form-group">
                <label className="form-label">Total Bilirubin (mg/dL)</label>
                <input
                  type="number"
                  step="0.1"
                  className="form-input"
                  value={totalBilirubin}
                  onChange={e => setTotalBilirubin(e.target.value)}
                  placeholder="e.g. 1.0"
                />
              </div>

              <div className="form-group" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                <label className="form-label" style={{ marginBottom: '0.4rem' }}>AST / ALT Elevated (&gt;3x ULN)</label>
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', fontSize: '0.85rem', color: '#334155' }}>
                  <input
                    type="checkbox"
                    checked={astAltElevated}
                    onChange={e => setAstAltElevated(e.target.checked)}
                    style={{ width: '16px', height: '16px', accentColor: '#4f46e5' }}
                  />
                  <span>Significant Transaminitis</span>
                </label>
              </div>
            </div>
          </div>

          {/* Safety Checklists: Allergies, Resistance, Medications, Prior Exposure */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '1rem', marginBottom: '1.25rem' }}>
            {/* Allergies */}
            <div style={{ border: '1px solid #fee2e2', background: '#fff5f5', padding: '0.85rem', borderRadius: '8px' }}>
              <div style={{ fontWeight: 700, fontSize: '0.82rem', color: '#991b1b', marginBottom: '0.5rem' }}>
                ⚠️ Documented Drug Allergies
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                {ALLERGY_OPTIONS.map(alg => (
                  <label key={alg} style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', fontSize: '0.8rem', color: '#450a0a', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={drugAllergies.includes(alg)}
                      onChange={() => toggleArrayItem(alg, drugAllergies, setDrugAllergies)}
                      style={{ accentColor: '#dc2626' }}
                    />
                    <span>{alg}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Resistance Profile */}
            <div style={{ border: '1px solid #fef3c7', background: '#fffbeb', padding: '0.85rem', borderRadius: '8px' }}>
              <div style={{ fontWeight: 700, fontSize: '0.82rem', color: '#92400e', marginBottom: '0.5rem' }}>
                🛡️ Prior Resistance Profile
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                {RESISTANCE_PROFILE_OPTIONS.map(res => (
                  <label key={res} style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', fontSize: '0.8rem', color: '#78350f', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={resistanceProfile.includes(res)}
                      onChange={() => toggleArrayItem(res, resistanceProfile, setResistanceProfile)}
                      style={{ accentColor: '#d97706' }}
                    />
                    <span>{res}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Current Medications */}
            <div style={{ border: '1px solid #e0e7ff', background: '#f5f7ff', padding: '0.85rem', borderRadius: '8px' }}>
              <div style={{ fontWeight: 700, fontSize: '0.82rem', color: '#3730a3', marginBottom: '0.5rem' }}>
                💊 Current Concurrent Medications
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                {MEDICATION_OPTIONS.map(med => (
                  <label key={med} style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', fontSize: '0.8rem', color: '#1e1b4b', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={currentMedications.includes(med)}
                      onChange={() => toggleArrayItem(med, currentMedications, setCurrentMedications)}
                      style={{ accentColor: '#4f46e5' }}
                    />
                    <span>{med}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Prior Antibiotic Exposure */}
            <div style={{ border: '1px solid #e2e8f0', background: '#f8fafc', padding: '0.85rem', borderRadius: '8px' }}>
              <div style={{ fontWeight: 700, fontSize: '0.82rem', color: '#334155', marginBottom: '0.5rem' }}>
                ⏳ Prior Exposure (&lt;90 days)
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                {PREVIOUS_EXPOSURE_OPTIONS.map(exp => (
                  <label key={exp} style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', fontSize: '0.8rem', color: '#334155', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={previousExposure.includes(exp)}
                      onChange={() => toggleArrayItem(exp, previousExposure, setPreviousExposure)}
                      style={{ accentColor: '#64748b' }}
                    />
                    <span>{exp}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>

          <button
            className="btn btn-primary"
            onClick={handleRunPatientSuitability}
            disabled={loading}
            style={{ width: '100%', padding: '0.85rem', fontWeight: 700, fontSize: '0.95rem' }}
          >
            {loading ? (
              <>
                <Spinner size={16} /> Evaluating Candidate Suitability & TreeSHAP…
              </>
            ) : (
              '⚡ Predict & Rank Antimicrobial Suitability'
            )}
          </button>
        </div>
      )}

      {/* Mode 2: In-Vitro Direct AST */}
      {activeMode === 'in_vitro_direct' && (
        <div className="card" style={{ marginBottom: '1.5rem', background: '#ffffff', border: '1px solid #e2e8f0' }}>
          <h3 className="card-title" style={{ margin: 0, marginBottom: '1rem', color: '#0f172a' }}>
            🔬 Direct In-Vitro Susceptibility Phenotype Tester
          </h3>
          <div className="form-grid" style={{ marginBottom: '1.25rem' }}>
            <div className="form-group">
              <label className="form-label">Identified Microorganism</label>
              <select
                className="form-select"
                value={identifiedOrganism}
                onChange={e => setIdentifiedOrganism(e.target.value)}
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
                value={directAntibiotic}
                onChange={e => setDirectAntibiotic(e.target.value)}
              >
                {COMMON_ANTIBIOTICS.map(abx => (
                  <option key={abx} value={abx}>
                    {abx}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Specimen Source</label>
              <select
                className="form-select"
                value={infectionSite}
                onChange={e => setInfectionSite(e.target.value)}
              >
                {INFECTION_SITES.map(site => (
                  <option key={site} value={site}>
                    {site}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <button
            className="btn btn-primary"
            onClick={handleRunDirectInVitro}
            disabled={loading}
            style={{ width: '100%', padding: '0.75rem', fontWeight: 600 }}
          >
            {loading ? <><Spinner size={16} /> Evaluating Phenotype…</> : '⚡ Run In-Vitro Prediction'}
          </button>
        </div>
      )}

      {error && <ErrorBox message={error} onRetry={activeMode === 'patient_specific' ? handleRunPatientSuitability : handleRunDirectInVitro} />}

      {/* ── Mode 1 Output: Patient-Specific Ranked Candidates ── */}
      {suitabilityResponse && (
        <div style={{ marginBottom: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
            <div>
              <h3 style={{ margin: 0, fontSize: '1.2rem', fontWeight: 700, color: '#0f172a' }}>
                📋 Candidate Antibiotic Ranking & Clinical Suitability
              </h3>
              <p style={{ margin: '0.2rem 0 0', fontSize: '0.82rem', color: '#64748b' }}>
                Ranked for <em>{suitabilityResponse.identified_organism}</em> ({suitabilityResponse.infection_site} isolate) in patient with eGFR: {String(suitabilityResponse.patient_summary.egfr ?? '')} mL/min.
              </p>
            </div>
            <span style={{ fontSize: '0.78rem', background: '#ecfdf5', color: '#047857', border: '1px solid #a7f3d0', padding: '0.3rem 0.75rem', borderRadius: '6px', fontWeight: 700 }}>
              {suitabilityResponse.ranked_candidates.length} Antimicrobials Evaluated
            </span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
            {suitabilityResponse.ranked_candidates.map((candidate: CandidateAntibioticRanking) => {
              const badge = getSusceptibilityBadge(candidate.predicted_susceptibility);
              const scoreColor = getScoreColor(candidate.suitability_score);
              const isExpanded = expandedCandidate === candidate.antibiotic;

              return (
                <div
                  key={candidate.antibiotic}
                  className="card"
                  style={{
                    padding: '1.1rem',
                    border: candidate.allergy_contraindication
                      ? '1px solid #fecaca'
                      : candidate.rank === 1
                      ? '2px solid #4f46e5'
                      : '1px solid #e2e8f0',
                    background: candidate.allergy_contraindication ? '#fffbfb' : '#ffffff',
                    transition: 'box-shadow 0.2s ease',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.75rem' }}>
                    {/* Left: Rank & Drug Title */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
                      <span
                        style={{
                          width: '34px',
                          height: '34px',
                          borderRadius: '50%',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontWeight: 800,
                          fontSize: '0.95rem',
                          background: candidate.rank === 1 ? '#4f46e5' : '#f1f5f9',
                          color: candidate.rank === 1 ? '#ffffff' : '#475569',
                        }}
                      >
                        #{candidate.rank}
                      </span>
                      <div>
                        <div style={{ fontSize: '1.05rem', fontWeight: 700, color: '#0f172a' }}>
                          {candidate.antibiotic}
                        </div>
                        <div style={{ fontSize: '0.78rem', color: '#64748b' }}>
                          Stewardship: <strong>{candidate.stewardship_tier}</strong>
                        </div>
                      </div>
                    </div>

                    {/* Middle: Badges & Resistance Prob */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap' }}>
                      <span
                        style={{
                          fontSize: '0.78rem',
                          fontWeight: 700,
                          padding: '0.25rem 0.65rem',
                          borderRadius: '6px',
                          background: badge.bg,
                          color: badge.color,
                          border: `1px solid ${badge.border}`,
                        }}
                      >
                        {candidate.predicted_susceptibility} ({(candidate.susceptibility_probability * 100).toFixed(0)}%)
                      </span>

                      <span
                        style={{
                          fontSize: '0.78rem',
                          padding: '0.25rem 0.65rem',
                          borderRadius: '6px',
                          background: '#f8fafc',
                          color: '#475569',
                          border: '1px solid #e2e8f0',
                        }}
                      >
                        P(Resist): <strong>{(candidate.resistance_probability * 100).toFixed(1)}%</strong>
                      </span>

                      {candidate.allergy_contraindication && (
                        <span style={{ fontSize: '0.75rem', fontWeight: 700, background: '#fee2e2', color: '#991b1b', padding: '0.25rem 0.6rem', borderRadius: '4px' }}>
                          ⚠️ CONTRAINDICATED
                        </span>
                      )}
                    </div>

                    {/* Right: Suitability Score Gauge */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
                      <div style={{ textAlign: 'right' }}>
                        <div style={{ fontSize: '0.72rem', color: '#64748b', textTransform: 'uppercase', fontWeight: 600 }}>
                          Suitability Score
                        </div>
                        <div style={{ fontSize: '1.25rem', fontWeight: 800, color: scoreColor }}>
                          {candidate.suitability_score.toFixed(0)}%
                        </div>
                      </div>

                      <button
                        className="btn btn-outline btn-sm"
                        onClick={() => setExpandedCandidate(isExpanded ? null : candidate.antibiotic)}
                        style={{ padding: '0.35rem 0.75rem', fontSize: '0.78rem' }}
                      >
                        {isExpanded ? 'Hide Details ▲' : 'View Safety & SHAP ▼'}
                      </button>
                    </div>
                  </div>

                  {/* Expanded Detail Panel */}
                  {isExpanded && (
                    <div style={{ marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid #e2e8f0', background: '#f8fafc', padding: '1rem', borderRadius: '8px' }}>
                      {/* Clinical Warnings */}
                      <div style={{ marginBottom: '1rem' }}>
                        <h5 style={{ margin: '0 0 0.5rem', fontSize: '0.82rem', fontWeight: 700, color: '#334155', textTransform: 'uppercase' }}>
                          Safety & Interaction Alerts
                        </h5>
                        {candidate.warnings.length === 0 ? (
                          <div style={{ fontSize: '0.82rem', color: '#047857', fontWeight: 500 }}>
                            ✓ No specific contraindications, dose modifications, or high-risk interactions identified.
                          </div>
                        ) : (
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                            {candidate.warnings.map((w, idx) => (
                              <div
                                key={idx}
                                style={{
                                  fontSize: '0.82rem',
                                  padding: '0.45rem 0.75rem',
                                  borderRadius: '6px',
                                  background: w.includes('CONTRAINDICATION') ? '#fee2e2' : '#fffbeb',
                                  color: w.includes('CONTRAINDICATION') ? '#991b1b' : '#92400e',
                                  border: w.includes('CONTRAINDICATION') ? '1px solid #fecaca' : '1px solid #fde68a',
                                  lineHeight: 1.4,
                                }}
                              >
                                {w}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>

                      {/* TreeSHAP Local Attributions */}
                      {candidate.shap_values && candidate.shap_values.length > 0 && (
                        <div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                            <h5 style={{ margin: 0, fontSize: '0.82rem', fontWeight: 700, color: '#334155', textTransform: 'uppercase' }}>
                              TreeSHAP Attribution (Impact on Resistance Odds)
                            </h5>
                            <span style={{ fontSize: '0.72rem', color: '#64748b' }}>ARMD-ECUH Trained TreeExplainer</span>
                          </div>
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                            {candidate.shap_values.map((item, sIdx) => {
                              const val = item.shap_value;
                              const isPos = val >= 0;
                              return (
                                <div key={sIdx} style={{ display: 'flex', alignItems: 'center', fontSize: '0.78rem' }}>
                                  <span style={{ width: '140px', fontWeight: 500, color: '#475569' }}>
                                    {item.feature}:
                                  </span>
                                  <div style={{ flex: 1, height: '10px', background: '#e2e8f0', borderRadius: '4px', overflow: 'hidden' }}>
                                    <div
                                      style={{
                                        width: `${Math.min(Math.abs(val) * 60, 100)}%`,
                                        height: '100%',
                                        background: isPos ? '#ef4444' : '#10b981',
                                      }}
                                    />
                                  </div>
                                  <span style={{ width: '55px', textAlign: 'right', fontWeight: 700, color: isPos ? '#dc2626' : '#059669', fontSize: '0.75rem' }}>
                                    {isPos ? `+${val.toFixed(2)}` : val.toFixed(2)}
                                  </span>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── Mode 2 Output: Single In-Vitro AST Result ── */}
      {directResponse && (
        <div className="card" style={{ marginBottom: '1.5rem', background: '#ffffff', border: '1px solid #e2e8f0' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <h3 className="card-title" style={{ margin: 0 }}>
              📊 In-Vitro Phenotypic Prediction
            </h3>
            <span
              style={{
                fontSize: '0.85rem',
                fontWeight: 700,
                padding: '0.35rem 0.85rem',
                borderRadius: '6px',
                ...getSusceptibilityBadge(directResponse.predicted_class),
              }}
            >
              {directResponse.predicted_class.toUpperCase()}
            </span>
          </div>

          <div style={{ background: '#f8fafc', padding: '1rem', borderRadius: '8px', marginBottom: '1.25rem' }}>
            <div style={{ fontSize: '0.85rem', color: '#64748b' }}>Tested Drug / Pathogen Combination:</div>
            <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#1e293b', marginTop: '0.2rem' }}>
              <em>{directResponse.organism}</em> vs. {directResponse.antibiotic}
            </div>
            <div style={{ fontSize: '0.8rem', color: '#64748b', marginTop: '0.2rem' }}>
              Specimen: {directResponse.culture_description} | Setting: {directResponse.ordering_mode}
            </div>
          </div>

          {/* Probabilities */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginBottom: '1rem' }}>
            {['Susceptible', 'Resistant', 'Intermediate'].map(cls => {
              const prob = directResponse.probabilities[cls] ?? 0;
              const barColor = cls === 'Susceptible' ? '#10b981' : cls === 'Resistant' ? '#ef4444' : '#f59e0b';
              return (
                <div key={cls}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', marginBottom: '0.2rem' }}>
                    <span>{cls}</span>
                    <span style={{ fontWeight: 600 }}>{(prob * 100).toFixed(1)}%</span>
                  </div>
                  <div style={{ height: '8px', background: '#e2e8f0', borderRadius: '9999px', overflow: 'hidden' }}>
                    <div style={{ width: `${Math.round(prob * 100)}%`, height: '100%', background: barColor }} />
                  </div>
                </div>
              );
            })}
          </div>

          <div style={{ fontSize: '0.78rem', color: '#64748b', background: '#f8fafc', padding: '0.75rem', borderRadius: '6px' }}>
            Validation: Tested on held-out patient encounters (ARMD-ECUH). Balanced Accuracy: 71.86%, Macro ROC-AUC: 0.8727.
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
