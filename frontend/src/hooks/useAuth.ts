/**
 * useAuth hook — manages JWT token in localStorage, login/logout state.
 */
import { useState, useCallback } from 'react';
import { login as apiLogin } from '../api/endpoints';
import { extractErrorMessage } from '../api/client';

interface AuthState {
  token: string | null;
  role: string | null;
  username: string | null;
}

export function useAuth() {
  const [auth, setAuth] = useState<AuthState>(() => ({
    token: localStorage.getItem('access_token'),
    role: localStorage.getItem('user_role'),
    username: localStorage.getItem('username'),
  }));
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const login = useCallback(async (username: string, password: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiLogin({ username, password });
      localStorage.setItem('access_token', res.access_token);
      localStorage.setItem('user_role', res.role);
      localStorage.setItem('username', username);
      setAuth({ token: res.access_token, role: res.role, username });
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('user_role');
    localStorage.removeItem('username');
    setAuth({ token: null, role: null, username: null });
  }, []);

  return {
    isAuthenticated: Boolean(auth.token),
    role: auth.role,
    username: auth.username,
    login,
    logout,
    loading,
    error,
  };
}
