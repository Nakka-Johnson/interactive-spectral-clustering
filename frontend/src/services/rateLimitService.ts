/**
 * PHASE 9: Frontend Rate Limiting & Error Handling
 * 
 * Handles 429 responses, implements exponential backoff,
 * and provides user feedback for rate limiting scenarios.
 */

// Rate limiting utilities
export class RateLimitHandler {
  private retryAttempts: Map<string, number> = new Map();
  private lastRetryTime: Map<string, number> = new Map();
  
  /**
   * Handle 429 rate limit response with exponential backoff
   */
  async handleRateLimit(
    response: Response, 
    requestKey: string, 
    maxRetries: number = 3
  ): Promise<Response | null> {
    
    const retryAfter = this.getRetryAfter(response);
    const currentAttempts = this.retryAttempts.get(requestKey) || 0;
    
    if (currentAttempts >= maxRetries) {
      this.showRateLimitMessage("Too many requests. Please try again later.");
      return null;
    }
    
    // Calculate backoff delay
    const baseDelay = retryAfter || Math.pow(2, currentAttempts) * 1000;
    const jitter = Math.random() * 1000; // Add jitter to prevent thundering herd
    const delay = baseDelay + jitter;
    
    // Update retry tracking
    this.retryAttempts.set(requestKey, currentAttempts + 1);
    this.lastRetryTime.set(requestKey, Date.now());
    
    // Show user feedback
    this.showRetryMessage(delay);
    
    // Wait and retry
    await this.sleep(delay);
    
    return null; // Caller should retry the original request
  }
  
  /**
   * Extract retry-after header value
   */
  private getRetryAfter(response: Response): number | null {
    const retryAfter = response.headers.get('Retry-After');
    if (!retryAfter) return null;
    
    // Handle both seconds and date formats
    const seconds = parseInt(retryAfter);
    return isNaN(seconds) ? null : seconds * 1000;
  }
  
  /**
   * Check if we should attempt a request (not in backoff period)
   */
  canMakeRequest(requestKey: string): boolean {
    const lastRetry = this.lastRetryTime.get(requestKey);
    if (!lastRetry) return true;
    
    const minInterval = 1000; // Minimum 1 second between attempts
    return Date.now() - lastRetry > minInterval;
  }
  
  /**
   * Reset retry attempts for successful requests
   */
  resetRetries(requestKey: string): void {
    this.retryAttempts.delete(requestKey);
    this.lastRetryTime.delete(requestKey);
  }
  
  /**
   * Show rate limit message to user
   */
  private showRateLimitMessage(message: string): void {
    // Integration with your notification system
    console.warn('[Rate Limit]', message);
    
    // Show toast notification
    if (typeof window !== 'undefined' && (window as any).showToast) {
      (window as any).showToast({
        type: 'warning',
        title: 'Rate Limit Exceeded',
        message: message,
        duration: 5000
      });
    }
  }
  
  /**
   * Show retry message to user
   */
  private showRetryMessage(delay: number): void {
    const seconds = Math.ceil(delay / 1000);
    const message = `Request limit reached. Retrying in ${seconds} seconds...`;
    
    console.info('[Rate Limit]', message);
    
    if (typeof window !== 'undefined' && (window as any).showToast) {
      (window as any).showToast({
        type: 'info',
        title: 'Retrying Request',
        message: message,
        duration: delay
      });
    }
  }
  
  /**
   * Sleep utility
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Enhanced API client with rate limiting support
export class ApiClient {
  private rateLimitHandler = new RateLimitHandler();
  private baseURL: string;
  
  constructor(baseURL: string = 'http://localhost:8002') {
    this.baseURL = baseURL;
  }
  
  /**
   * Make API request with automatic rate limiting handling
   */
  async request<T>(
    endpoint: string, 
    options: RequestInit = {},
    maxRetries: number = 3
  ): Promise<T> {
    
    const requestKey = `${options.method || 'GET'}:${endpoint}`;
    const url = `${this.baseURL}${endpoint}`;
    
    // Check if we can make the request (not in backoff)
    if (!this.rateLimitHandler.canMakeRequest(requestKey)) {
      throw new Error('Request blocked due to rate limiting backoff');
    }
    
    // Add rate limit headers for tracking
    const headers = {
      'Content-Type': 'application/json',
      ...options.headers,
    };
    
    try {
      const response = await fetch(url, {
        ...options,
        headers
      });
      
      // Handle rate limiting
      if (response.status === 429) {
        const retryResponse = await this.rateLimitHandler.handleRateLimit(
          response, 
          requestKey, 
          maxRetries
        );
        
        if (retryResponse) {
          return retryResponse.json();
        } else {
          // Retry the original request
          return this.request(endpoint, options, maxRetries - 1);
        }
      }
      
      // Handle other errors
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new ApiError(response.status, errorData.message || 'Request failed', errorData);
      }
      
      // Success - reset retry attempts
      this.rateLimitHandler.resetRetries(requestKey);
      
      return response.json();
      
    } catch (error) {
      if (error instanceof ApiError) {
        throw error;
      }
      
      // Network or other errors
      console.error('[API Error]', error);
      throw new ApiError(0, 'Network error occurred', { originalError: error });
    }
  }
  
  // Convenience methods
  async get<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'GET' });
  }
  
  async post<T>(endpoint: string, data: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: JSON.stringify(data)
    });
  }
  
  async put<T>(endpoint: string, data: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'PUT',
      body: JSON.stringify(data)
    });
  }
  
  async delete<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'DELETE' });
  }
}

// Custom error class for API errors
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public data?: any
  ) {
    super(message);
    this.name = 'ApiError';
  }
  
  get isRateLimit(): boolean {
    return this.status === 429;
  }
  
  get isServerError(): boolean {
    return this.status >= 500;
  }
  
  get isClientError(): boolean {
    return this.status >= 400 && this.status < 500;
  }
}

// Rate limit context interface
export interface RateLimitState {
  isRateLimited: boolean;
  retryAfter: number | null;
  handleRateLimit: (seconds: number) => void;
}

// Export singleton instance
export const apiClient = new ApiClient();
export const rateLimitHandler = new RateLimitHandler();
