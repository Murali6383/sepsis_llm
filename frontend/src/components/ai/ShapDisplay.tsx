/**
 * SHAP Explanation Display Component.
 * Renders top contributing features for a model prediction.
 * Shows both positive (risk-increasing) and negative (risk-decreasing) contributions.
 */
import React from 'react';
import type { ShapValue } from '../../types';

interface ShapDisplayProps {
  shapValues: ShapValue[];
  title?: string;
}

export function ShapDisplay({ shapValues, title = 'Top Feature Contributions (SHAP)' }: ShapDisplayProps) {
  if (!shapValues || shapValues.length === 0) {
    return (
      <div style={{ color: 'var(--color-text-muted)', fontSize: '0.85rem' }}>
        No SHAP values available for this prediction.
      </div>
    );
  }

  const maxAbs = Math.max(...shapValues.map(sv => Math.abs(sv.shap_value)), 0.001);

  return (
    <div>
      <p style={{ fontSize: '0.8rem', color: 'var(--color-text-muted)', marginBottom: '0.75rem' }}>
        🔴 Red = increases risk &nbsp;|&nbsp; 🟢 Green = decreases risk
        &nbsp;|&nbsp; Width = relative magnitude
      </p>
      <div className="shap-list">
        {shapValues.map((sv, i) => {
          const isPositive = sv.shap_value > 0;
          const width = Math.abs(sv.shap_value / maxAbs) * 100;
          return (
            <div key={i} className="shap-item">
              <span className="shap-feature-name" title={sv.feature}>
                {sv.feature}
              </span>
              <div className="shap-bar-container">
                <div
                  className={isPositive ? 'shap-bar-positive' : 'shap-bar-negative'}
                  style={{ width: `${width}%` }}
                />
              </div>
              <span className="shap-value" style={{ color: isPositive ? 'var(--color-danger)' : 'var(--color-success)' }}>
                {sv.shap_value > 0 ? '+' : ''}{sv.shap_value.toFixed(4)}
              </span>
              {sv.value !== null && (
                <span style={{ color: 'var(--color-text-muted)', fontSize: '0.78rem', width: '60px', textAlign: 'right' }}>
                  val={typeof sv.value === 'number' ? sv.value.toFixed(2) : sv.value}
                </span>
              )}
            </div>
          );
        })}
      </div>
      <p style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', marginTop: '0.5rem' }}>
        SHAP values reflect statistical associations in synthetic training data — not clinical causation.
      </p>
    </div>
  );
}
