/**
 * Authentication & Navigation Test Suite for SepsisGuardian AI.
 *
 * Verifies all 6 required test cases:
 * 1. Login with valid credentials → Dashboard opens immediately.
 * 2. Login with invalid credentials → Error message appears.
 * 3. Refresh dashboard → User remains authenticated.
 * 4. Logout → Login page opens immediately.
 * 5. Access dashboard without authentication → Redirect to login.
 * 6. Open login page while already authenticated → Redirect to dashboard.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';
import { MemoryRouter, Routes, Route, Navigate } from 'react-router-dom';

vi.mock('../api/endpoints', () => ({
  login: vi.fn(),
  getAIHealth: vi.fn().mockResolvedValue({ llm_configured: false }),
  predictSepsis: vi.fn(),
  predictAKI: vi.fn(),
  predictCardio: vi.fn(),
  predictLiver: vi.fn(),
  predictNeuro: vi.fn(),
  predictResp: vi.fn(),
  predictAntibiotic: vi.fn(),
  predictPatientAntibioticSuitability: vi.fn(),
  explainPrediction: vi.fn(),
  chatWithAI: vi.fn(),
  generateReport: vi.fn(),
}));

import * as endpoints from '../api/endpoints';
import { AuthProvider, useAuth } from '../context/AuthContext';
import { LoginPage } from '../pages/LoginPage';
import { DashboardPage } from '../pages/DashboardPage';
import { ProtectedRoute } from '../components/auth/ProtectedRoute';

function TestApp({ initialRoute = '/login' }: { initialRoute?: string }) {
  return (
    <MemoryRouter initialEntries={[initialRoute]}>
      <AuthProvider>
        <TestRoutes />
      </AuthProvider>
    </MemoryRouter>
  );
}

function TestRoutes() {
  const { isAuthenticated, username, role, logout } = useAuth();

  return (
    <Routes>
      <Route
        path="/login"
        element={isAuthenticated ? <Navigate to="/" replace /> : <LoginPage />}
      />
      <Route
        path="/"
        element={
          <ProtectedRoute>
            <DashboardPage username={username} role={role} onLogout={logout} />
          </ProtectedRoute>
        }
      />
      <Route
        path="*"
        element={<Navigate to={isAuthenticated ? '/' : '/login'} replace />}
      />
    </Routes>
  );
}

describe('Authentication Flow & Navigation Tests', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
  });

  it('1. Login with valid credentials → Dashboard opens immediately without manual refresh', async () => {
    (endpoints.login as ReturnType<typeof vi.fn>).mockResolvedValue({
      access_token: 'mock-valid-jwt-token-12345',
      token_type: 'bearer',
      role: 'doctor',
    });

    render(<TestApp initialRoute="/login" />);

    // Check we start at LoginPage
    expect(screen.getByRole('button', { name: /Log In/i })).toBeInTheDocument();

    // Fill credentials
    fireEvent.change(screen.getByPlaceholderText(/doctor \/ nurse/i), {
      target: { value: 'doctor' },
    });
    fireEvent.change(screen.getByPlaceholderText(/••••••••/i), {
      target: { value: 'doctor123' },
    });

    // Submit
    fireEvent.click(screen.getByRole('button', { name: /Log In/i }));

    // Assert dashboard opens immediately
    await waitFor(() => {
      expect(screen.getByText(/Sepsis Risk & Organ Dysfunction Predictions/i)).toBeInTheDocument();
      expect(screen.getAllByText('doctor').length).toBeGreaterThanOrEqual(1);
    });

    // Verify token was stored in localStorage
    expect(localStorage.getItem('access_token')).toBe('mock-valid-jwt-token-12345');
    expect(localStorage.getItem('user_role')).toBe('doctor');
    expect(localStorage.getItem('username')).toBe('doctor');
  });

  it('2. Login with invalid credentials → Error message appears and does not redirect', async () => {
    (endpoints.login as ReturnType<typeof vi.fn>).mockRejectedValue({
      response: {
        status: 401,
        data: { detail: 'Incorrect username or password' },
      },
      isAxiosError: true,
    });

    render(<TestApp initialRoute="/login" />);

    fireEvent.change(screen.getByPlaceholderText(/doctor \/ nurse/i), {
      target: { value: 'wrong_user' },
    });
    fireEvent.change(screen.getByPlaceholderText(/••••••••/i), {
      target: { value: 'wrong_pass' },
    });

    fireEvent.click(screen.getByRole('button', { name: /Log In/i }));

    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent(/Incorrect username or password/i);
    });

    // Still on Login page
    expect(screen.getByRole('button', { name: /Log In/i })).toBeInTheDocument();
    expect(localStorage.getItem('access_token')).toBeNull();
  });

  it('3. Refresh dashboard → User remains authenticated from valid session', async () => {
    // Pre-seed localStorage as if user logged in previously
    localStorage.setItem('access_token', 'persisted-jwt-token-abc');
    localStorage.setItem('user_role', 'nurse');
    localStorage.setItem('username', 'nurse');

    // Mount at dashboard route /
    render(<TestApp initialRoute="/" />);

    // Dashboard should load immediately without redirecting to login
    await waitFor(() => {
      expect(screen.getByText(/Sepsis Risk & Organ Dysfunction Predictions/i)).toBeInTheDocument();
      expect(screen.getByText('nurse')).toBeInTheDocument();
    });
  });

  it('4. Logout → Login page opens immediately and clears state', async () => {
    localStorage.setItem('access_token', 'persisted-jwt-token-abc');
    localStorage.setItem('user_role', 'doctor');
    localStorage.setItem('username', 'doctor');

    render(<TestApp initialRoute="/" />);

    await waitFor(() => {
      expect(screen.getByText(/Log Out/i)).toBeInTheDocument();
    });

    // Click Log Out
    fireEvent.click(screen.getByText(/Log Out/i));

    // Must immediately redirect to LoginPage
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Log In/i })).toBeInTheDocument();
    });

    // Verify localStorage cleared
    expect(localStorage.getItem('access_token')).toBeNull();
    expect(localStorage.getItem('user_role')).toBeNull();
    expect(localStorage.getItem('username')).toBeNull();
  });

  it('5. Access dashboard without authentication → Redirect to login', async () => {
    // Empty localStorage
    render(<TestApp initialRoute="/" />);

    // Should be redirected to /login
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Log In/i })).toBeInTheDocument();
      expect(screen.queryByText(/Sepsis Risk & Organ Dysfunction Predictions/i)).not.toBeInTheDocument();
    });
  });

  it('6. Open login page while already authenticated → Redirect to dashboard', async () => {
    localStorage.setItem('access_token', 'valid-active-session');
    localStorage.setItem('user_role', 'doctor');
    localStorage.setItem('username', 'doctor');

    render(<TestApp initialRoute="/login" />);

    // Should redirect to dashboard immediately
    await waitFor(() => {
      expect(screen.getByText(/Sepsis Risk & Organ Dysfunction Predictions/i)).toBeInTheDocument();
      expect(screen.queryByRole('button', { name: /Log In/i })).not.toBeInTheDocument();
    });
  });
});
