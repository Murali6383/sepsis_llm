/**
 * Root App component — handles auth routing.
 */
import React from 'react';
import { useAuth } from './hooks/useAuth';
import { LoginPage } from './pages/LoginPage';
import { DashboardPage } from './pages/DashboardPage';

export default function App() {
  const { isAuthenticated, username, role, logout } = useAuth();

  if (!isAuthenticated) {
    return <LoginPage />;
  }

  return (
    <DashboardPage
      username={username}
      role={role}
      onLogout={logout}
    />
  );
}
