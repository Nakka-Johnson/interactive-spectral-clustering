import React, { useState } from 'react';
import { useToast, useToastUtils } from '../components/ToastProvider';
import { RetryButton, NetworkRetryButton, ErrorRetryCard } from '../components/RetryButton';

/**
 * Demo component for PHASE 7 Error Handling Features
 * 
 * This component demonstrates all the error handling and notification
 * features implemented for production readiness.
 */
export function ErrorHandlingDemo() {
  const toast = useToast();
  const toastUtils = useToastUtils();
  const [isLoading, setIsLoading] = useState(false);
  const [hasError, setHasError] = useState(false);

  // Simulate different error scenarios
  const simulateNetworkError = async () => {
    setIsLoading(true);
    try {
      // Simulate network failure
      await new Promise((_, reject) => 
        setTimeout(() => reject(new Error('Network connection failed')), 1000)
      );
    } catch (error) {
      toastUtils.networkError(() => simulateNetworkError());
      setHasError(true);
    } finally {
      setIsLoading(false);
    }
  };

  const simulateValidationError = () => {
    toastUtils.validationError('Please provide a valid file format (.csv or .xlsx)');
  };

  const simulateClusteringSuccess = () => {
    toastUtils.clusteringSuccess('Your data has been processed with 3 clusters identified.');
  };

  const simulateClusteringError = () => {
    toastUtils.clusteringError(
      'Clustering failed due to insufficient data points. Please upload a larger dataset.',
      () => {
        toast.info('Retrying clustering...', 'Attempting to process with different parameters.');
        setTimeout(() => simulateClusteringSuccess(), 2000);
      }
    );
  };

  const simulateUploadError = () => {
    toastUtils.uploadError('File size exceeds 10MB limit. Please compress your data or split into smaller files.');
  };

  const simulateUploadSuccess = () => {
    toastUtils.uploadSuccess('customer_data.csv');
  };

  const triggerJavaScriptError = () => {
    // This will be caught by the ErrorBoundary
    throw new Error('Simulated component crash for testing error boundary');
  };

  const simulateServerError = async () => {
    setIsLoading(true);
    try {
      // Simulate server error
      const response = await fetch('/api/nonexistent-endpoint');
      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }
    } catch (error) {
      toast.error(
        'Server Error',
        'The server is currently unavailable. Please try again later.',
        {
          persistent: true,
          action: {
            label: 'Retry',
            onClick: () => simulateServerError()
          }
        }
      );
      setHasError(true);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-8">
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h1 className="text-2xl font-bold text-gray-900 mb-2">
          PHASE 7: Error Handling Demo
        </h1>
        <p className="text-gray-600 mb-6">
          This page demonstrates the production-ready error handling and notification system.
        </p>

        {/* Toast Notifications Section */}
        <section className="mb-8">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Toast Notifications
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <button
              onClick={() => toast.success('Success!', 'Operation completed successfully.')}
              className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors"
            >
              Success Toast
            </button>
            <button
              onClick={() => toast.error('Error!', 'Something went wrong.')}
              className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors"
            >
              Error Toast
            </button>
            <button
              onClick={() => toast.warning('Warning!', 'Please review your input.')}
              className="px-4 py-2 bg-yellow-600 text-white rounded-md hover:bg-yellow-700 transition-colors"
            >
              Warning Toast
            </button>
            <button
              onClick={() => toast.info('Info', 'Here\'s some helpful information.')}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
            >
              Info Toast
            </button>
          </div>
        </section>

        {/* Application-Specific Toasts */}
        <section className="mb-8">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Application-Specific Notifications
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            <button
              onClick={simulateValidationError}
              className="px-4 py-2 bg-orange-600 text-white rounded-md hover:bg-orange-700 transition-colors"
            >
              Validation Error
            </button>
            <button
              onClick={simulateClusteringSuccess}
              className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors"
            >
              Clustering Success
            </button>
            <button
              onClick={simulateClusteringError}
              className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors"
            >
              Clustering Error
            </button>
            <button
              onClick={simulateUploadSuccess}
              className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors"
            >
              Upload Success
            </button>
            <button
              onClick={simulateUploadError}
              className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors"
            >
              Upload Error
            </button>
            <button
              onClick={simulateNetworkError}
              className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors"
            >
              Network Error
            </button>
          </div>
        </section>

        {/* Retry Buttons Section */}
        <section className="mb-8">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Retry Button Components
          </h2>
          <div className="space-y-4">
            <div className="flex items-center space-x-4">
              <RetryButton
                onRetry={async () => {
                  await new Promise(resolve => setTimeout(resolve, 2000));
                  toast.success('Retry successful!');
                }}
                size="sm"
              >
                Small Retry
              </RetryButton>
              
              <RetryButton
                onRetry={async () => {
                  await new Promise(resolve => setTimeout(resolve, 2000));
                  toast.success('Retry successful!');
                }}
                size="md"
                variant="secondary"
              >
                Medium Retry
              </RetryButton>
              
              <RetryButton
                onRetry={async () => {
                  await new Promise(resolve => setTimeout(resolve, 2000));
                  toast.success('Retry successful!');
                }}
                size="lg"
                variant="outline"
              >
                Large Retry
              </RetryButton>
            </div>

            <div>
              <h3 className="text-sm font-medium text-gray-700 mb-2">Network-Aware Retry Button:</h3>
              <NetworkRetryButton
                onRetry={async () => {
                  await new Promise(resolve => setTimeout(resolve, 1500));
                  toast.success('Network operation successful!');
                }}
                showNetworkStatus={true}
              >
                Network Retry
              </NetworkRetryButton>
            </div>
          </div>
        </section>

        {/* Error Cards Section */}
        <section className="mb-8">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Error Display Components
          </h2>
          
          {hasError && (
            <ErrorRetryCard
              title="Network Connection Failed"
              message="Unable to connect to the clustering service. Please check your internet connection and try again."
              onRetry={async () => {
                setHasError(false);
                await new Promise(resolve => setTimeout(resolve, 1000));
                toast.success('Connection restored!');
              }}
              retryButtonText="Reconnect"
              showDetails={true}
              errorDetails={`Network error occurred at ${new Date().toISOString()}\nStatus: Connection timeout\nEndpoint: /api/cluster`}
              className="mb-4"
            />
          )}
          
          <button
            onClick={() => setHasError(true)}
            className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors"
          >
            Show Error Card
          </button>
        </section>

        {/* Error Boundary Test */}
        <section className="mb-8">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Error Boundary Test
          </h2>
          <p className="text-sm text-gray-600 mb-4">
            This button will trigger a JavaScript error that will be caught by the global error boundary.
            The error boundary will display a fallback UI with recovery options.
          </p>
          <button
            onClick={triggerJavaScriptError}
            className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors"
          >
            Trigger Component Crash
          </button>
        </section>

        {/* Server Error Test */}
        <section className="mb-8">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Server Error Handling
          </h2>
          <p className="text-sm text-gray-600 mb-4">
            This demonstrates how server errors are handled with persistent notifications and retry actions.
          </p>
          <button
            onClick={simulateServerError}
            disabled={isLoading}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            {isLoading ? 'Connecting...' : 'Test Server Error'}
          </button>
        </section>

        {/* Clear All Toasts */}
        <section>
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Toast Management
          </h2>
          <button
            onClick={() => toast.clearAllToasts()}
            className="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 transition-colors"
          >
            Clear All Toasts
          </button>
        </section>
      </div>
    </div>
  );
}

export default ErrorHandlingDemo;
