/**
 * Research Report Generator Component.
 * Generates a structured research report using actual model outputs via POST /api/ai/report
 * and exports publication-grade clinical PDFs via POST /api/ai/report/pdf.
 * Does not invent values or fabricate clinical measurements.
 */
import React, { useState } from 'react';
import type { PatientContext } from '../../types';
import { generateReport, downloadPDFReport } from '../../api/endpoints';
import { extractErrorMessage } from '../../api/client';
import { Spinner, ErrorBox, LLMTag, Disclaimer } from '../shared';

interface ReportGeneratorProps {
  patientContext: PatientContext;
  llmConfigured: boolean;
}

export function ReportGenerator({ patientContext, llmConfigured }: ReportGeneratorProps) {
  const [report, setReport] = useState<{ text: string; generatedAt: string; disclaimer: string } | null>(null);
  const [loading, setLoading] = useState(false);
  const [downloadingPdf, setDownloadingPdf] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pdfError, setPdfError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const hasContext = Boolean(
    patientContext.sepsis_result ||
    (patientContext.organ_results && patientContext.organ_results.length > 0) ||
    patientContext.antibiotic_result
  );

  const handleGenerate = async () => {
    if (loading) return;
    setLoading(true);
    setError(null);

    try {
      const resp = await generateReport({
        sepsis_result: patientContext.sepsis_result
          ? {
              sepsis_probability: patientContext.sepsis_result.sepsis_probability,
              sepsis_label: patientContext.sepsis_result.sepsis_label,
              risk_level: patientContext.sepsis_result.risk_level,
              model_type: patientContext.sepsis_result.model_type,
            }
          : undefined,
        organ_results: patientContext.organ_results?.map(o => ({
          organ: o.organ,
          probability: o.probability,
          label: o.label,
          risk_level: o.risk_level,
        })),
        antibiotic_result: patientContext.antibiotic_result
          ? {
              predicted_class: patientContext.antibiotic_result.predicted_class,
              organism: patientContext.antibiotic_result.organism,
              antibiotic: patientContext.antibiotic_result.antibiotic,
              data_disclaimer: patientContext.antibiotic_result.data_disclaimer,
            }
          : undefined,
        shap_values: patientContext.sepsis_result?.shap_values?.slice(0, 8) ?? [],
        patient_ref: 'PT-ANON-DEMO',
      });
      setReport({
        text: resp.report_text,
        generatedAt: new Date(resp.generated_at).toLocaleString(),
        disclaimer: resp.disclaimer,
      });
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadPdf = async () => {
    if (downloadingPdf) return;
    setDownloadingPdf(true);
    setPdfError(null);

    try {
      const blob = await downloadPDFReport({
        sepsis_result: patientContext.sepsis_result
          ? {
              sepsis_probability: patientContext.sepsis_result.sepsis_probability,
              sepsis_label: patientContext.sepsis_result.sepsis_label,
              risk_level: patientContext.sepsis_result.risk_level,
              model_type: patientContext.sepsis_result.model_type,
            }
          : undefined,
        organ_results: patientContext.organ_results?.map(o => ({
          organ: o.organ,
          probability: o.probability,
          label: o.label,
          risk_level: o.risk_level,
        })),
        antibiotic_result: patientContext.antibiotic_result
          ? {
              predicted_class: patientContext.antibiotic_result.predicted_class,
              probabilities: patientContext.antibiotic_result.probabilities,
              organism: patientContext.antibiotic_result.organism,
              antibiotic: patientContext.antibiotic_result.antibiotic,
              culture_description: patientContext.antibiotic_result.culture_description,
              ordering_mode: patientContext.antibiotic_result.ordering_mode,
              data_disclaimer: patientContext.antibiotic_result.data_disclaimer,
            }
          : undefined,
        shap_values: [
          ...(patientContext.sepsis_result?.shap_values?.slice(0, 5) ?? []),
          ...(patientContext.antibiotic_result?.shap_values?.slice(0, 4) ?? []),
        ],
        patient_ref: 'PT-CLINICAL-SUMMARY',
      });

      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `SepsisGuardian_Report_${Date.now()}.pdf`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      setPdfError(extractErrorMessage(err));
    } finally {
      setDownloadingPdf(false);
    }
  };

  const handleCopy = () => {
    if (!report) return;
    navigator.clipboard.writeText(report.text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const handleDownloadTxt = () => {
    if (!report) return;
    const blob = new Blob([report.text + '\n\n' + report.disclaimer], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `sepsis_guardian_report_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div>
      <Disclaimer text="Report is compiled from verified model outputs and empirical ARMD-ECUH microbiology evaluations. For research demonstration only." />

      {!hasContext && (
        <div style={{ color: '#64748b', fontSize: '0.875rem', marginBottom: '1.25rem', background: '#f8fafc', padding: '0.75rem', borderRadius: '6px' }}>
          ℹ️ Please run sepsis risk evaluation or ARMD-ECUH antimicrobial susceptibility prediction to populate the report data.
        </div>
      )}

      {/* Action Buttons Row */}
      <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', marginBottom: '1.25rem' }}>
        <button
          className="btn btn-primary"
          onClick={handleGenerate}
          disabled={loading || !hasContext || !llmConfigured}
          style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
        >
          {loading ? (
            <>
              <Spinner size={16} /> Synthesizing LLM Narrative…
            </>
          ) : (
            '📝 Generate Research Report'
          )}
        </button>

        <button
          className="btn"
          onClick={handleDownloadPdf}
          disabled={downloadingPdf || !hasContext}
          style={{
            background: '#0284c7',
            color: '#ffffff',
            border: 'none',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            padding: '0.5rem 1rem',
            borderRadius: '6px',
            fontWeight: 600,
            cursor: downloadingPdf || !hasContext ? 'not-allowed' : 'pointer',
            opacity: downloadingPdf || !hasContext ? 0.6 : 1,
          }}
        >
          {downloadingPdf ? (
            <>
              <Spinner size={16} /> Generating PDF…
            </>
          ) : (
            '📥 Download Publication-Grade PDF'
          )}
        </button>
      </div>

      {!llmConfigured && (
        <div style={{ fontSize: '0.8rem', color: '#b45309', background: '#fffbeb', padding: '0.5rem 0.75rem', borderRadius: '6px', marginBottom: '1rem', border: '1px solid #fde68a' }}>
          ⚠️ Groq LLM not configured in backend/.env — text report requires Groq, but PDF export with empirical model tables remains fully functional.
        </div>
      )}

      {error && <div style={{ marginBottom: '1rem' }}><ErrorBox message={error} onRetry={handleGenerate} /></div>}
      {pdfError && <div style={{ marginBottom: '1rem' }}><ErrorBox message={pdfError} onRetry={handleDownloadPdf} /></div>}

      {/* Generated Report View */}
      {report && (
        <div style={{ marginTop: '1.25rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.75rem', flexWrap: 'wrap' }}>
            <LLMTag />
            <span style={{ fontSize: '0.8rem', color: '#64748b' }}>
              Generated: {report.generatedAt}
            </span>
            <button className="btn btn-outline btn-sm" onClick={handleCopy}>
              {copied ? '✓ Copied' : '📋 Copy Text'}
            </button>
            <button className="btn btn-outline btn-sm" onClick={handleDownloadTxt}>
              ⬇ Download .txt
            </button>
          </div>

          <pre
            style={{
              background: '#f8fafc',
              borderRadius: '8px',
              padding: '1.25rem',
              whiteSpace: 'pre-wrap',
              fontSize: '0.85rem',
              lineHeight: 1.65,
              border: '1px solid #e2e8f0',
              maxHeight: '500px',
              overflowY: 'auto',
              color: '#1e293b',
              fontFamily: 'Consolas, Monaco, "Courier New", monospace',
            }}
          >
            {report.text}
          </pre>

          <div className="disclaimer" style={{ marginTop: '0.75rem' }}>
            {report.disclaimer}
          </div>
        </div>
      )}
    </div>
  );
}
