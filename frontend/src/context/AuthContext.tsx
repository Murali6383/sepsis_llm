/**
 * AuthContext — Application-wide Authentication Context.
 *
 * Solves the isolated-state bug by maintaining a single, shared authentication
 * state tree across the entire React component hierarchy.
 *
 * Synchronizes with localStorage and dispatches immediate state updates to
 * prevent race conditions and eliminate the need for manual browser reloads.
 */
import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';
import { login as apiLogin } from '../api/endpoints';
import { extractErrorMessage } from '../api/client';
import jwt_decode from 'jwt-decode';

export interface JWTTokenPayload {
  exp: number; // expiration timestamp (seconds since epoch)
  iat?: number;
  sub?: string;
  [key: string]: any;
}

export interface AuthContextType {
  token: string | null;
  role: string | null;
  username: string | null;
  isAuthenticated: boolean;
  loading: boolean;
  error: string | null;
  login: (username: string, password: string) => Promise<boolean>;
  logout: () => void;
  clearError: () => void;
}

export interface AuthContextType {
  token: string | null;
  role: string | null;
  username: string | null;
  isAuthenticated: boolean;
  loading: boolean;
  error: string | null;
  login: (username: string, password: string) => Promise<boolean>;
  logout: () => void;
  clearError: () => void;
}

export const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [token, setToken] = useState<string | null>(() => localStorage.getItem('access_token'));
  const [role, setRole] = useState<string | null>(() => localStorage.getItem('user_role'));
  const [username, setUsername] = useState<string | null>(() => localStorage.getItem('username'));
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Sync token state changes across browser tabs or windows
  useEffect(() => {
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'access_token') {
        setToken(e.newValue);
      } else if (e.key === 'user_role') {
        setRole(e.newValue);
      } else if (e.key === 'username') {
        setUsername(e.newValue);
      }
    };
    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('user_role');
    localStorage.removeItem('username');
    setToken(null);
    setRole(null);
    setUsername(null);
    setError(null);
  }, []);

  // Handle unauthorized events dispatched by API client interceptor
  useEffect(() => {
    const handleUnauthorized = () => {
      logout();
    };
    window.addEventListener('auth:unauthorized', handleUnauthorized);
    return () => window.removeEventListener('auth:unauthorized', handleUnauthorized);
  }, [logout]);

  const login = useCallback(async (user: string, pass: string): Promise<boolean> => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiLogin({ username: user, password: pass });

      // 1. Store authentication tokens securely in localStorage
      localStorage.setItem('access_token', res.access_token);
      localStorage.setItem('user_role', res.role);
      localStorage.setItem('username', user);

      // 2. Immediately update context state so subscriber components re-render instantaneously
      setToken(res.access_token);
      setRole(res.role);
      setUsername(user);
      setLoading(false);
      return true;
    } catch (err) {
      const msg = extractErrorMessage(err);
      setError(msg);
      setLoading(false);
      return false;
    }
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const value: AuthContextType = {
    token,
    role,
    username,
    isAuthenticated: Boolean(token),
    loading,
    error,
    login,
    logout,
    clearError,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
