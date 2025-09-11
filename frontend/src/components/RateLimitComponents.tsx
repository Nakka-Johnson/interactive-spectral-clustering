/**
 * PHASE 9: React Components for Rate Limiting UI
 */

import React, { useState, useCallback } from 'react';
import { RateLimitState } from '../services/rateLimitService';

// React hook for rate limit awareness
export function useRateLimitAware(): RateLimitState {
  const [isRateLimited, setIsRateLimited] = useState(false);
  const [retryAfter, setRetryAfter] = useState<number | null>(null);
  
  const handleRateLimit = useCallback((seconds: number) => {
    setIsRateLimited(true);
    setRetryAfter(seconds);
    
    // Auto-clear after retry period
    setTimeout(() => {
      setIsRateLimited(false);
      setRetryAfter(null);
    }, seconds * 1000);
  }, []);
  
  return {
    isRateLimited,
    retryAfter,
    handleRateLimit
  };
}

// Global rate limit status component
export function RateLimitStatus() {
  const { isRateLimited, retryAfter } = useRateLimitAware();
  
  if (!isRateLimited) return null;
  
  return (
    <div className="fixed top-0 left-0 right-0 z-50 bg-yellow-100 border-b border-yellow-400 px-4 py-3">
      <div className="flex items-center justify-center">
        <div className="flex items-center space-x-2">
          <svg className="w-5 h-5 text-yellow-600" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
          </svg>
          <span className="font-semibold text-yellow-800">Rate Limit Active</span>
          {retryAfter && (
            <span className="text-yellow-700"> - Please wait {retryAfter} seconds before retrying</span>
          )}
        </div>
      </div>
    </div>
  );
}

// Button component that respects rate limiting
interface RateLimitButtonProps {
  onClick: () => Promise<void> | void;
  children: React.ReactNode;
  disabled?: boolean;
  className?: string;
  rateLimitKey?: string;
}

export function RateLimitButton({ 
  onClick, 
  children, 
  disabled = false, 
  className = '',
  rateLimitKey 
}: RateLimitButtonProps) {
  const [isLoading, setIsLoading] = useState(false);
  const [rateLimited, setRateLimited] = useState(false);
  const [retrySeconds, setRetrySeconds] = useState(0);
  
  const handleClick = async () => {
    if (isLoading || rateLimited) return;
    
    setIsLoading(true);
    try {
      await onClick();
    } catch (error: any) {
      // Handle rate limiting error
      if (error?.status === 429) {
        const retryAfter = error?.data?.retry_after || 60;
        setRateLimited(true);
        setRetrySeconds(retryAfter);
        
        // Countdown timer
        const timer = setInterval(() => {
          setRetrySeconds(prev => {
            if (prev <= 1) {
              clearInterval(timer);
              setRateLimited(false);
              return 0;
            }
            return prev - 1;
          });
        }, 1000);
      }
      throw error;
    } finally {
      setIsLoading(false);
    }
  };
  
  const buttonDisabled = disabled || isLoading || rateLimited;
  const buttonText = rateLimited 
    ? `Retry in ${retrySeconds}s` 
    : isLoading 
      ? 'Loading...' 
      : children;
  
  return (
    <button
      onClick={handleClick}
      disabled={buttonDisabled}
      className={`
        px-4 py-2 rounded font-medium transition-colors
        ${buttonDisabled 
          ? 'bg-gray-300 text-gray-500 cursor-not-allowed' 
          : 'bg-blue-600 text-white hover:bg-blue-700'
        }
        ${className}
      `}
    >
      {buttonText}
    </button>
  );
}

// Form component with rate limiting
interface RateLimitFormProps {
  onSubmit: (data: any) => Promise<void>;
  children: React.ReactNode;
  className?: string;
  rateLimitKey?: string;
}

export function RateLimitForm({ 
  onSubmit, 
  children, 
  className = '',
  rateLimitKey 
}: RateLimitFormProps) {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [rateLimited, setRateLimited] = useState(false);
  const [retryAfter, setRetryAfter] = useState(0);
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (isSubmitting || rateLimited) return;
    
    setIsSubmitting(true);
    
    try {
      const formData = new FormData(e.target as HTMLFormElement);
      const data = Object.fromEntries(Array.from(formData.entries()));
      await onSubmit(data);
    } catch (error: any) {
      if (error?.status === 429) {
        const seconds = error?.data?.retry_after || 60;
        setRateLimited(true);
        setRetryAfter(seconds);
        
        setTimeout(() => {
          setRateLimited(false);
          setRetryAfter(0);
        }, seconds * 1000);
      }
      throw error;
    } finally {
      setIsSubmitting(false);
    }
  };
  
  return (
    <form onSubmit={handleSubmit} className={className}>
      {children}
      {rateLimited && (
        <div className="mt-4 p-3 bg-yellow-100 border border-yellow-400 rounded">
          <p className="text-yellow-800 text-sm">
            Rate limit exceeded. Please wait {retryAfter} seconds before retrying.
          </p>
        </div>
      )}
    </form>
  );
}

// Hook for handling API calls with rate limiting
export function useApiWithRateLimit() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [rateLimited, setRateLimited] = useState(false);
  
  const makeRequest = useCallback(async (requestFn: () => Promise<any>) => {
    setIsLoading(true);
    setError(null);
    setRateLimited(false);
    
    try {
      const result = await requestFn();
      return result;
    } catch (error: any) {
      if (error?.status === 429) {
        setRateLimited(true);
        setError('Rate limit exceeded. Please try again later.');
      } else {
        setError(error?.message || 'An error occurred');
      }
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, []);
  
  return {
    isLoading,
    error,
    rateLimited,
    makeRequest
  };
}

const RateLimitComponents = {
  RateLimitStatus,
  RateLimitButton,
  RateLimitForm,
  useRateLimitAware,
  useApiWithRateLimit
};

export default RateLimitComponents;
