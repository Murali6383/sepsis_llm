/**
 * Shared utility components.
 */
import React from 'react';
import type { RiskLevel } from '../../types';

// ── Spinner ───────────────────────────────────────────────────────────────────
export function Spinner({ size = 20 }: { size?: number }) {
  return (
    <div
      className="spinner"
      style={{ width: size, height: size }}
      role="status"
      aria-label="Loading"
    />
  );
}

// ── Risk Badge ────────────────────────────────────────────────────────────────
export function RiskBadge({ level }: { level: RiskLevel }) {
  const cls =
    level === 'high' ? 'badge-danger' :
    level === 'moderate' ? 'badge-warning' : 'badge-success';
  return <span className={`badge ${cls}`}>{level.toUpperCase()}</span>;
}

// ── Risk Bar ──────────────────────────────────────────────────────────────────
export function RiskBar({ probability, level }: { probability: number; level: RiskLevel }) {
  const fillClass =
    level === 'high' ? 'risk-high' :
    level === 'moderate' ? 'risk-moderate' : 'risk-low';
  return (
    <div className="risk-bar" role="progressbar" aria-valuenow={Math.round(probability * 100)} aria-valuemin={0} aria-valuemax={100}>
      <div
        className={`risk-bar-fill ${fillClass}`}
        style={{ width: `${Math.round(probability * 100)}%` }}
      />
    </div>
  );
}

// ── Error Box ─────────────────────────────────────────────────────────────────
export function ErrorBox({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="error-box" role="alert">
      <strong>Error:</strong> {message}
      {onRetry && (
        <button className="btn btn-outline btn-sm" style={{ marginLeft: '0.75rem' }} onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

// ── Disclaimer ────────────────────────────────────────────────────────────────
export function Disclaimer({ text }: { text?: string }) {
  return (
    <div className="disclaimer" role="note">
      ⚠️ {text ?? 'All datasets are synthetic. For research demonstration only. Not a clinical tool.'}
    </div>
  );
}

// ── LLM Tag ───────────────────────────────────────────────────────────────────
export function LLMTag() {
  return (
    <span className="badge badge-info" style={{ fontSize: '0.7rem' }}>
      🤖 LLM-generated
    </span>
  );
}

// ── Section Header ────────────────────────────────────────────────────────────
export function SectionHeader({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <div style={{ marginBottom: '1.5rem' }}>
      <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--color-text)' }}>{title}</h2>
      {subtitle && <p style={{ color: 'var(--color-text-muted)', fontSize: '0.875rem', marginTop: '0.25rem' }}>{subtitle}</p>}
    </div>
  );
}
