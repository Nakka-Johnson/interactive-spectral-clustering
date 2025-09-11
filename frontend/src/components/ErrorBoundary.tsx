import React, { Component, ErrorInfo, ReactNode } from 'react';
import { WarningAmber, Refresh, Home, BugReport } from '@mui/icons-material';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  errorId: string;
}

/**
 * Global Error Boundary Component for PHASE 7 - Fixed for Phase 4
 * 
 * Catches JavaScript errors anywhere in the component tree and displays
 * a fallback UI with recovery options and error reporting.
 * 
 * Features:
 * - Graceful error handling with user-friendly messages
 * - Error reporting with unique error IDs
 * - Recovery actions (retry, reload, home)
 * - Development mode detailed error display
 * - Production mode safe error messages
 */
export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      errorId: ''
    };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    // Update state so the next render will show the fallback UI
    const errorId = `ERR-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    return {
      hasError: true,
      error,
      errorId
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Log error details
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    
    this.setState({
      error,
      errorInfo
    });

    // Call custom error handler if provided
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    // Report error to monitoring service (if available)
    this.reportError(error, errorInfo);
  }

  private reportError = async (error: Error, errorInfo: ErrorInfo) => {
    try {
      // Send error report to backend monitoring
      const errorReport = {
        message: error.message,
        stack: error.stack,
        componentStack: errorInfo.componentStack,
        errorId: this.state.errorId,
        timestamp: new Date().toISOString(),
        userAgent: navigator.userAgent,
        url: window.location.href
      };

      // Only report in production or if explicitly enabled
      if (process.env.NODE_ENV === 'production' || process.env.REACT_APP_ERROR_REPORTING === 'true') {
        await fetch('/api/errors/report', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(errorReport)
        }).catch(() => {
          // Silently fail error reporting to avoid recursive errors
        });
      }
    } catch (reportingError) {
      console.error('Failed to report error:', reportingError);
    }
  };

  private handleRetry = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      errorId: ''
    });
  };

  private handleReload = () => {
    window.location.reload();
  };

  private handleGoHome = () => {
    window.location.href = '/';
  };

  private handleReportBug = () => {
    const subject = encodeURIComponent(`Bug Report - Error ID: ${this.state.errorId}`);
    const body = encodeURIComponent(`
Error ID: ${this.state.errorId}
Error Message: ${this.state.error?.message || 'Unknown error'}
Timestamp: ${new Date().toISOString()}
URL: ${window.location.href}

Please describe what you were doing when this error occurred:

`);
    
    const mailtoUrl = `mailto:support@example.com?subject=${subject}&body=${body}`;
    window.open(mailtoUrl);
  };

  render() {
    if (this.state.hasError) {
      // Custom fallback UI if provided
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // Import Material-UI components directly here for easier replacement
      const { Box, Container, Paper, Typography, Button, Alert, Stack } = require('@mui/material');

      // Default error UI using Material-UI
      return (
        <Box
          sx={{
            minHeight: '100vh',
            bgcolor: 'grey.50',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            py: 6,
            px: 3,
          }}
        >
          <Container maxWidth="sm">
            <Paper sx={{ p: 4, textAlign: 'center' }}>
              {/* Error Icon */}
              <Box
                sx={{
                  mx: 'auto',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  height: 64,
                  width: 64,
                  borderRadius: '50%',
                  bgcolor: 'error.light',
                  mb: 2,
                }}
              >
                                <WarningAmber sx={{ fontSize: 32, color: '#d32f2f' }} />
              </Box>

              {/* Error Title */}
              <Typography variant="h5" component="h2" gutterBottom>
                Something went wrong
              </Typography>

              {/* Error Message */}
              <Typography variant="body2" color="text.secondary" gutterBottom>
                We're sorry, but something unexpected happened.
              </Typography>
              
              <Typography variant="caption" color="text.secondary" sx={{ mb: 3, display: 'block' }}>
                Error ID: <Box component="code" sx={{ bgcolor: 'grey.100', px: 0.5, borderRadius: 0.5 }}>{this.state.errorId}</Box>
              </Typography>

              {/* Development Mode Error Details */}
              {process.env.NODE_ENV === 'development' && this.state.error && (
                <Alert severity="error" sx={{ mt: 2, mb: 3, textAlign: 'left' }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Development Error Details:
                  </Typography>
                  <Box
                    component="pre"
                    sx={{
                      fontSize: '0.75rem',
                      whiteSpace: 'pre-wrap',
                      overflowX: 'auto',
                      maxHeight: 200,
                      overflow: 'auto',
                    }}
                  >
                    {this.state.error.message}
                    {this.state.error.stack && '\n\nStack trace:\n' + this.state.error.stack}
                  </Box>
                </Alert>
              )}

              {/* Recovery Actions */}
              <Stack spacing={2} sx={{ mt: 3 }}>
                <Button
                  onClick={this.handleRetry}
                  variant="contained"
                  startIcon={<Refresh />}
                  fullWidth
                >
                  Try Again
                </Button>

                <Stack direction="row" spacing={2}>
                  <Button
                    onClick={this.handleGoHome}
                    variant="outlined"
                    startIcon={<Home />}
                    fullWidth
                  >
                    Home
                  </Button>

                  <Button
                    onClick={this.handleReload}
                    variant="outlined"
                    startIcon={<Refresh />}
                    fullWidth
                  >
                    Reload
                  </Button>
                </Stack>

                <Button
                  onClick={this.handleReportBug}
                  variant="text"
                  startIcon={<BugReport />}
                  fullWidth
                  color="inherit"
                >
                  Report Bug
                </Button>
              </Stack>

              {/* Help Text */}
              <Typography variant="caption" color="text.secondary" sx={{ mt: 3, display: 'block' }}>
                If this problem persists, please contact our support team.
              </Typography>
            </Paper>
          </Container>
        </Box>
      );
    }

    return this.props.children;
  }
}

// Higher-Order Component for easier usage
export function withErrorBoundary<P extends object>(
  Component: React.ComponentType<P>,
  errorBoundaryProps?: Omit<Props, 'children'>
) {
  const WrappedComponent = (props: P) => (
    <ErrorBoundary {...errorBoundaryProps}>
      <Component {...props} />
    </ErrorBoundary>
  );

  WrappedComponent.displayName = `withErrorBoundary(${Component.displayName || Component.name})`;
  
  return WrappedComponent;
}

export default ErrorBoundary;
