/**
 * useAuth hook — re-exports unified auth context hook and types.
 * Preserves backward compatibility while guaranteeing single-source-of-truth state.
 */
export { useAuth, AuthProvider } from '../context/AuthContext';
export type { AuthContextType } from '../context/AuthContext';
