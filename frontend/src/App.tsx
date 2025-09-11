/**
 * Main App component for Interactive Spectral Clustering Platform.
 *
 * Provides routing, theme, and global layout structure.
 * Integrates with the Zustand store for state management.
 *
 * PHASE 7: Enhanced with global error boundary and toast notifications.
 */

import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, CssBaseline } from '@mui/material';
import { neoDarkTheme } from './theme/neoDarkTheme';
import { FileUpload } from './features/upload/FileUpload';
import { ExperimentDashboard } from './features/experiments/ExperimentDashboard';
import VisualizationPage from './features/visualize/VisualizationPage';
import { logger } from './utils/logger';
import { Phase2TestPage } from './pages/Phase2TestPage';
// import { GridSearchForm } from './features/params'; // Temporarily disabled
import { Leaderboard } from './features/experiments';
import { EmbeddingVisualizationPage } from './pages/EmbeddingVisualizationPage';
import { ErrorHandlingDemo } from './pages/ErrorHandlingDemo';
import ReportPage from './pages/ReportPage';
import { AppShell } from './components/layout';
import { useAppStore } from './store/appStore';

// PHASE 2 & 3: Import new upload and config components
import UploadPage from './features/upload/UploadPage';
import ConfigPage from './features/config/ConfigPage';

// PHASE 7: Import error handling components
import { ErrorBoundary } from './components/ErrorBoundary';
import { ToastProvider } from './components/ToastProvider';

// PHASE 7: Neo Dark Theme imported from ./theme/neoDarkTheme.ts

/**
 * Main App component with routing and layout.
 */
/**
 * Main clustering application component.
 */
const ClusteringApp: React.FC = () => {
  // Initialize store data on app load
  const { fetchDatasets, fetchRuns } = useAppStore();

  React.useEffect(() => {
    fetchDatasets();
    fetchRuns();
  }, [fetchDatasets, fetchRuns]); // Include dependencies

  return (
    <ThemeProvider theme={neoDarkTheme}>
      <CssBaseline />
      <Router>
        <AppShell>
          <Routes>
            {/* Upload Route - Phase 2 Enhanced */}
            <Route
              path="/upload"
              element={
                <UploadPage />
              }
            />

            {/* Data Explore Route */}
            <Route
              path="/data-explore"
              element={
                <ExperimentDashboard
                  onViewRun={(run) => {
                    logger.info('View run:', run);
                    // TODO: Navigate to visualization with run data
                  }}
                  onStartRun={() => {
                    logger.info('Start new run');
                    // TODO: Navigate to configuration page
                  }}
                  onDeleteRun={(runId) => {
                    logger.info('Delete run:', runId);
                    // TODO: Implement delete functionality
                  }}
                />
              }
            />

            {/* Config Route - Phase 3 Enhanced */}
            <Route
              path="/config"
              element={
                <ConfigPage />
              }
            />

            {/* Visualization Route */}
            <Route path="/visualize" element={<VisualizationPage />} />

            {/* Metrics Route */}
            <Route
              path="/metrics"
              element={
                <div style={{ padding: '20px' }}>
                  <h2>Metrics</h2>
                  <p>Performance metrics and analytics.</p>
                </div>
              }
            />

            {/* Big Data Route */}
            <Route
              path="/big-data"
              element={
                <div style={{ padding: '20px' }}>
                  <h2>Big Data</h2>
                  <p>Large-scale data processing and analysis.</p>
                </div>
              }
            />

            {/* Enterprise Route */}
            <Route
              path="/enterprise"
              element={
                <div style={{ padding: '20px' }}>
                  <h2>Enterprise</h2>
                  <p>Enterprise features and integrations.</p>
                </div>
              }
            />

            {/* History Route */}
            <Route
              path="/history"
              element={
                <div style={{ padding: '20px' }}>
                  <h2>History</h2>
                  <p>Analysis history and past runs.</p>
                </div>
              }
            />

            {/* Report Route */}
            <Route
              path="/report"
              element={
                <div style={{ padding: '20px' }}>
                  <h2>Report</h2>
                  <p>Generate and view analysis reports.</p>
                </div>
              }
            />

            {/* Legacy Routes */}
            <Route path="/experiments" element={<Navigate to="/data-explore" replace />} />

            {/* Embedding Visualization Route */}
            <Route path="/embedding" element={<EmbeddingVisualizationPage />} />

            {/* Grid Search Route - Temporarily disabled due to Material-UI Grid compatibility */}
            <Route
              path="/grid-search"
              element={
                <div style={{ padding: '20px', textAlign: 'center' }}>
                  <h2>Grid Search</h2>
                  <p>
                    Grid Search functionality is temporarily disabled for Material-UI compatibility
                    updates.
                  </p>
                </div>
              }
            />

            {/* Leaderboard Route */}
            <Route
              path="/leaderboard"
              element={
                <Leaderboard
                  onSelectRun={(runId) => {
                    logger.info('Selected run:', runId);
                    // TODO: Navigate to visualization with run data
                  }}
                  onRefresh={() => {
                    logger.info('Refreshing leaderboard...');
                    // TODO: Implement refresh functionality
                  }}
                />
              }
            />

            {/* PHASE 7: Error Handling Demo */}
            <Route path="/error-demo" element={<ErrorHandlingDemo />} />

            {/* Report Generation Page */}
            <Route path="/reports" element={<ReportPage />} />

            {/* PHASE 2 Test Route */}
            <Route path="/test-phase2" element={<Phase2TestPage />} />

            {/* Default redirect to upload */}
            <Route path="/" element={<Navigate to="/upload" replace />} />

            {/* Catch-all redirect */}
            <Route path="*" element={<Navigate to="/upload" replace />} />
          </Routes>
        </AppShell>
      </Router>
    </ThemeProvider>
  );
};

/**
 * Root App Component with Error Boundary and Toast Provider
 *
 * PHASE 7: Wraps the main app with global error handling and notifications.
 */
function App() {
  return (
    <ErrorBoundary
      onError={(error, errorInfo) => {
        // Log error to console for debugging
        logger.error('Global error caught:', error, errorInfo);

        // In production, you could send this to an error tracking service
        if (process.env.NODE_ENV === 'production') {
          // Example: Send to error tracking service
          // errorTrackingService.captureException(error, { extra: errorInfo });
        }
      }}
    >
      <ToastProvider>
        <ClusteringApp />
      </ToastProvider>
    </ErrorBoundary>
  );
}

export default App;
