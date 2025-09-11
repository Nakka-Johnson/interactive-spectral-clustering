import React, { useState } from 'react';
import { Refresh, Wifi, WifiOff, ErrorOutline } from '@mui/icons-material';

interface RetryButtonProps {
  onRetry: () => Promise<void> | void;
  disabled?: boolean;
  size?: 'sm' | 'md' | 'lg';
  variant?: 'primary' | 'secondary' | 'outline';
  showIcon?: boolean;
  loadingText?: string;
  children?: React.ReactNode;
  className?: string;
}

/**
 * Retry Button Component for PHASE 7
 *
 * A reusable button component for retry actions with loading states,
 * network status awareness, and different variants.
 */
export function RetryButton({
  onRetry,
  disabled = false,
  size = 'md',
  variant = 'primary',
  showIcon = true,
  loadingText = 'Retrying...',
  children = 'Retry',
  className = '',
}: RetryButtonProps) {
  const [isRetrying, setIsRetrying] = useState(false);
  const [retryCount, setRetryCount] = useState(0);

  const handleRetry = async () => {
    if (isRetrying || disabled) return;

    try {
      setIsRetrying(true);
      setRetryCount((prev) => prev + 1);
      await onRetry();
    } catch (error) {
      console.error('Retry failed:', error);
      // Error handling is handled by the parent component
    } finally {
      setIsRetrying(false);
    }
  };

  const getSizeClasses = () => {
    switch (size) {
      case 'sm':
        return 'px-3 py-1.5 text-sm';
      case 'lg':
        return 'px-6 py-3 text-lg';
      default:
        return 'px-4 py-2 text-base';
    }
  };

  const getVariantClasses = () => {
    const baseClasses =
      'font-medium rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 transition-all duration-200';

    switch (variant) {
      case 'secondary':
        return `${baseClasses} bg-gray-100 text-gray-900 hover:bg-gray-200 focus:ring-gray-500 disabled:bg-gray-50 disabled:text-gray-400`;
      case 'outline':
        return `${baseClasses} border border-gray-300 bg-white text-gray-700 hover:bg-gray-50 focus:ring-blue-500 disabled:border-gray-200 disabled:text-gray-400`;
      default:
        return `${baseClasses} bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500 disabled:bg-blue-300`;
    }
  };

  const getIconSize = () => {
    switch (size) {
      case 'sm':
        return 'h-3 w-3';
      case 'lg':
        return 'h-5 w-5';
      default:
        return 'h-4 w-4';
    }
  };

  const isDisabled = disabled || isRetrying;

  return (
    <button
      onClick={handleRetry}
      disabled={isDisabled}
      className={`
        inline-flex items-center justify-center
        ${getSizeClasses()}
        ${getVariantClasses()}
        ${isDisabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'}
        ${className}
      `}
      title={retryCount > 0 ? `Retry attempt ${retryCount + 1}` : 'Retry'}
    >
      {showIcon && (
        <Refresh
          className={`
            ${getIconSize()}
            ${children ? 'mr-2' : ''}
            ${isRetrying ? 'animate-spin' : ''}
          `}
        />
      )}
      {isRetrying ? loadingText : children}
    </button>
  );
}

interface NetworkRetryButtonProps extends Omit<RetryButtonProps, 'onRetry'> {
  onRetry: () => Promise<void> | void;
  showNetworkStatus?: boolean;
}

/**
 * Network-aware Retry Button
 *
 * A retry button that shows network status and is disabled when offline.
 */
export function NetworkRetryButton({
  onRetry,
  showNetworkStatus = true,
  ...props
}: NetworkRetryButtonProps) {
  const [isOnline, setIsOnline] = useState(navigator.onLine);

  React.useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  return (
    <div className="flex items-center space-x-2">
      <RetryButton {...props} onRetry={onRetry} disabled={props.disabled || !isOnline} />

      {showNetworkStatus && (
        <div className="flex items-center text-sm text-gray-500">
          {isOnline ? (
            <div className="flex items-center text-green-600">
              <Wifi className="h-4 w-4 mr-1" />
              <span>Online</span>
            </div>
          ) : (
            <div className="flex items-center text-red-600">
              <WifiOff className="h-4 w-4 mr-1" />
              <span>Offline</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

interface ErrorRetryCardProps {
  title: string;
  message: string;
  onRetry: () => Promise<void> | void;
  retryButtonText?: string;
  showDetails?: boolean;
  errorDetails?: string;
  className?: string;
}

/**
 * Error Retry Card Component
 *
 * A complete error display with retry functionality.
 */
export function ErrorRetryCard({
  title,
  message,
  onRetry,
  retryButtonText = 'Try Again',
  showDetails = false,
  errorDetails,
  className = '',
}: ErrorRetryCardProps) {
  const [showErrorDetails, setShowErrorDetails] = useState(false);

  return (
    <div className={`bg-red-50 border border-red-200 rounded-lg p-6 ${className}`}>
      <div className="flex">
        <div className="flex-shrink-0">
          <ErrorOutline className="h-5 w-5 text-red-400" />
        </div>
        <div className="ml-3 flex-1">
          <h3 className="text-sm font-medium text-red-800">{title}</h3>
          <div className="mt-2 text-sm text-red-700">
            <p>{message}</p>
          </div>

          {showDetails && errorDetails && (
            <div className="mt-4">
              <button
                onClick={() => setShowErrorDetails(!showErrorDetails)}
                className="text-sm text-red-600 hover:text-red-500 underline focus:outline-none"
              >
                {showErrorDetails ? 'Hide' : 'Show'} error details
              </button>

              {showErrorDetails && (
                <div className="mt-2 p-3 bg-red-100 rounded border text-xs text-red-800 font-mono whitespace-pre-wrap">
                  {errorDetails}
                </div>
              )}
            </div>
          )}

          <div className="mt-4">
            <RetryButton onRetry={onRetry} size="sm">
              {retryButtonText}
            </RetryButton>
          </div>
        </div>
      </div>
    </div>
  );
}

export default RetryButton;
