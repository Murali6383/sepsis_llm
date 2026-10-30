/**
 * Axios API client — all requests go through /api proxy to FastAPI backend.
 * The Groq API key NEVER touches this client or the browser.
 */
import axios, { AxiosError } from 'axios';

// Base URL: in development Vite proxies /api → http://localhost:8000
const api = axios.create({
  baseURL: '/',
  timeout: 35_000,
  headers: { 'Content-Type': 'application/json' },
});

// Inject auth token from localStorage on every request
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Safe error extraction — never expose stack traces to UI
export function extractErrorMessage(err: unknown): string {
  if (err instanceof AxiosError) {
    const detail = err.response?.data?.detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) return detail.map((d: { msg?: string }) => d.msg ?? String(d)).join('; ');
    if (err.response?.status === 401) return 'Authentication required. Please log in.';
    if (err.response?.status === 503) return 'Service unavailable. Check backend or Groq configuration.';
    if (err.code === 'ECONNABORTED') return 'Request timed out. Please try again.';
  }
  return 'An unexpected error occurred.';
}

export default api;
