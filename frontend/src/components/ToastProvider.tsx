import React, { createContext, useContext, useCallback, useState, useEffect } from 'react';
import { Close } from '@mui/icons-material';

export type ToastType = 'success' | 'error' | 'warning' | 'info';

export interface Toast {
  id: string;
  type: ToastType;
  title: string;
  message?: string;
  duration?: number;
  persistent?: boolean;
  action?: {
    label: string;
    onClick: () => void;
  };
}

interface ToastContextType {
  toasts: Toast[];
  addToast: (toast: Omit<Toast, 'id'>) => string;
  removeToast: (id: string) => void;
  clearAllToasts: () => void;
  success: (title: string, message?: string, options?: Partial<Toast>) => string;
  error: (title: string, message?: string, options?: Partial<Toast>) => string;
  warning: (title: string, message?: string, options?: Partial<Toast>) => string;
  info: (title: string, message?: string, options?: Partial<Toast>) => string;
}

const ToastContext = createContext<ToastContextType | undefined>(undefined);

/**
 * Toast Provider Component for PHASE 7
 * 
 * Provides a global toast notification system for user feedback.
 * Supports different toast types, auto-dismiss, persistent toasts,
 * and custom actions.
 */
export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const removeToast = useCallback((id: string) => {
    setToasts(prev => prev.filter(toast => toast.id !== id));
  }, []);

  const addToast = useCallback((toast: Omit<Toast, 'id'>) => {
    const id = `toast-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const newToast = {
      id,
      duration: 5000, // Default 5 seconds
      ...toast,
    };

    setToasts(prev => [...prev, newToast]);

    // Auto-remove toast after duration (unless persistent)
    if (!newToast.persistent && newToast.duration && newToast.duration > 0) {
      setTimeout(() => {
        removeToast(id);
      }, newToast.duration);
    }

    return id;
  }, [removeToast]);

  const clearAllToasts = useCallback(() => {
    setToasts([]);
  }, []);

  // Convenience methods for different toast types
  const success = useCallback((title: string, message?: string, options?: Partial<Toast>) => {
    return addToast({ type: 'success', title, message, ...options });
  }, [addToast]);

  const error = useCallback((title: string, message?: string, options?: Partial<Toast>) => {
    return addToast({ 
      type: 'error', 
      title, 
      message, 
      duration: 8000, // Errors stay longer
      ...options 
    });
  }, [addToast]);

  const warning = useCallback((title: string, message?: string, options?: Partial<Toast>) => {
    return addToast({ 
      type: 'warning', 
      title, 
      message, 
      duration: 6000,
      ...options 
    });
  }, [addToast]);

  const info = useCallback((title: string, message?: string, options?: Partial<Toast>) => {
    return addToast({ type: 'info', title, message, ...options });
  }, [addToast]);

  const contextValue: ToastContextType = {
    toasts,
    addToast,
    removeToast,
    clearAllToasts,
    success,
    error,
    warning,
    info,
  };

  return (
    <ToastContext.Provider value={contextValue}>
      {children}
      <ToastContainer />
    </ToastContext.Provider>
  );
}

/**
 * Toast Container Component
 * 
 * Renders all active toasts in a fixed position overlay.
 */
function ToastContainer() {
  const context = useContext(ToastContext);
  if (!context) return null;

  const { toasts, removeToast } = context;

  if (toasts.length === 0) return null;

  const { Box } = require('@mui/material');

  return (
    <Box
      sx={{
        position: 'fixed',
        top: 16,
        right: 16,
        zIndex: 1300, // Material-UI z-index for overlays
        display: 'flex',
        flexDirection: 'column',
        gap: 1.5,
        maxWidth: 400,
        width: '100%',
      }}
    >
      {toasts.map(toast => (
        <ToastItem
          key={toast.id}
          toast={toast}
          onRemove={() => removeToast(toast.id)}
        />
      ))}
    </Box>
  );
}

/**
 * Individual Toast Item Component
 */
function ToastItem({ toast, onRemove }: { toast: Toast; onRemove: () => void }) {
  const [isVisible, setIsVisible] = useState(false);
  const [isLeaving, setIsLeaving] = useState(false);

  useEffect(() => {
    // Animate in
    const timer = setTimeout(() => setIsVisible(true), 50);
    return () => clearTimeout(timer);
  }, []);

  const handleRemove = useCallback(() => {
    setIsLeaving(true);
    setTimeout(onRemove, 200); // Match animation duration
  }, [onRemove]);

  const getAlertSeverity = (): 'success' | 'error' | 'warning' | 'info' => {
    switch (toast.type) {
      case 'success':
        return 'success';
      case 'error':
        return 'error';
      case 'warning':
        return 'warning';
      case 'info':
      default:
        return 'info';
    }
  };

  const { Alert, AlertTitle, IconButton, Button, Box, Collapse } = require('@mui/material');

  return (
    <Collapse in={isVisible && !isLeaving} timeout={200}>
      <Alert
        severity={getAlertSeverity()}
        variant="filled"
        sx={{
          maxWidth: '100%',
          '& .MuiAlert-action': {
            alignItems: 'flex-start',
            paddingTop: 0,
          },
        }}
        action={
          <IconButton
            size="small"
            aria-label="close"
            color="inherit"
            onClick={handleRemove}
          >
            <Close sx={{ fontSize: 16 }} />
          </IconButton>
        }
      >
        <AlertTitle sx={{ fontWeight: 600 }}>
          {toast.title}
        </AlertTitle>
        
        {toast.message && (
          <Box sx={{ mt: 0.5 }}>
            {toast.message}
          </Box>
        )}
        
        {toast.action && (
          <Box sx={{ mt: 1 }}>
            <Button
              size="small"
              color="inherit"
              variant="outlined"
              onClick={toast.action.onClick}
              sx={{
                borderColor: 'currentColor',
                '&:hover': {
                  borderColor: 'currentColor',
                  backgroundColor: 'rgba(255, 255, 255, 0.1)',
                },
              }}
            >
              {toast.action.label}
            </Button>
          </Box>
        )}
      </Alert>
    </Collapse>
  );
}

/**
 * Hook to use toast notifications
 */
export function useToast() {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider');
  }
  return context;
}

/**
 * Hook to use toast notifications with utility functions
 */
export function useToastUtils() {
  const toastContext = useToast();
  return createToastUtils(toastContext);
}

/**
 * Create toast utility functions that take a toast context
 */
export function createToastUtils(toastContext: ToastContextType) {
  return {
    /**
     * Show network error toast with retry action
     */
    networkError: (retryFn?: () => void) => {
      return toastContext.error(
        'Network Error',
        'Unable to connect to the server. Please check your connection.',
        {
          persistent: true,
          action: retryFn ? {
            label: 'Retry',
            onClick: retryFn
          } : undefined
        }
      );
    },

    /**
     * Show validation error toast
     */
    validationError: (message: string) => {
      return toastContext.warning('Validation Error', message);
    },

    /**
     * Show clustering operation success
     */
    clusteringSuccess: (message?: string) => {
      return toastContext.success(
        'Clustering Complete',
        message || 'Your clustering analysis has been completed successfully.'
      );
    },

    /**
     * Show clustering operation error
     */
    clusteringError: (message: string, retryFn?: () => void) => {
      return toastContext.error(
        'Clustering Failed',
        message,
        {
          action: retryFn ? {
            label: 'Try Again',
            onClick: retryFn
          } : undefined
        }
      );
    },

    /**
     * Show file upload error
     */
    uploadError: (message: string) => {
      return toastContext.error('Upload Failed', message);
    },

    /**
     * Show file upload success
     */
    uploadSuccess: (filename: string) => {
      return toastContext.success('File Uploaded', `${filename} has been uploaded successfully.`);
    }
  };
}

export default ToastProvider;
