/**
 * Login Page
 */
import React, { useState } from 'react';
import { useAuth } from '../hooks/useAuth';

export function LoginPage() {
  const { login, loading, error } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    login(username, password);
  };

  return (
    <div style={{
      minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'var(--color-bg)',
    }}>
      <div style={{
        background: 'var(--color-surface)', border: '1px solid var(--color-border)',
        borderRadius: '12px', padding: '2.5rem', width: '360px', boxShadow: 'var(--shadow)',
      }}>
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>🛡️</div>
          <h1 style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--color-primary)' }}>
            SepsisGuardian AI
          </h1>
          <p style={{ color: 'var(--color-text-muted)', fontSize: '0.85rem', marginTop: '0.25rem' }}>
            Research Demonstration Platform
          </p>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label className="form-label">Username</label>
            <input
              type="text"
              className="form-input"
              value={username}
              onChange={e => setUsername(e.target.value)}
              placeholder="doctor / nurse / admin"
              required
              autoFocus
            />
          </div>
          <div className="form-group">
            <label className="form-label">Password</label>
            <input
              type="password"
              className="form-input"
              value={password}
              onChange={e => setPassword(e.target.value)}
              placeholder="••••••••"
              required
            />
          </div>

          {error && (
            <div className="error-box" style={{ marginBottom: '1rem' }}>{error}</div>
          )}

          <button
            type="submit"
            className="btn btn-primary"
            style={{ width: '100%', justifyContent: 'center', padding: '0.65rem' }}
            disabled={loading}
          >
            {loading ? 'Logging in…' : 'Log In'}
          </button>
        </form>

        <div style={{ marginTop: '1.25rem', padding: '0.75rem', background: 'var(--color-surface-2)', borderRadius: '8px', fontSize: '0.8rem' }}>
          <p style={{ color: 'var(--color-text-muted)', marginBottom: '0.4rem' }}>Demo accounts:</p>
          <p>doctor / doctor123 &nbsp;|&nbsp; nurse / nurse123</p>
        </div>

        <div className="disclaimer" style={{ marginTop: '1rem', fontSize: '0.75rem' }}>
          ⚠️ All data is synthetic. For research demonstration only.
        </div>
      </div>
    </div>
  );
}
