import axios, { AxiosResponse } from 'axios';
import { io, Socket } from 'socket.io-client';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const WS_BASE_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 minutes for long clustering operations
  headers: {
    'Content-Type': 'application/json',
  },
});

// JWT token management
let authToken: string | null = null;

export const setAuthToken = (token: string | null) => {
  authToken = token;
  if (token) {
    apiClient.defaults.headers.common['Authorization'] = `Bearer ${token}`;
  } else {
    delete apiClient.defaults.headers.common['Authorization'];
  }
};

// Request interceptor for logging and JWT
apiClient.interceptors.request.use(
  (config) => {
    // Add JWT token if available
    if (authToken && !config.headers.Authorization) {
      config.headers.Authorization = `Bearer ${authToken}`;
    }
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling and toasts
apiClient.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('API Response Error:', error.response?.data || error.message);
    
    // Handle common error scenarios
    if (error.response?.status === 401) {
      // Unauthorized - clear token and redirect to login
      setAuthToken(null);
      // Could dispatch a toast here for unauthorized access
    } else if (error.response?.status >= 500) {
      // Server error - show user-friendly message
      console.error('Server error:', error.response.data);
    } else if (!error.response) {
      // Network error
      console.error('Network error - API may be unavailable');
    }
    
    return Promise.reject(error);
  }
);

// Type definitions
export interface UploadResponse {
  job_id: string;
  columns: string[];
  numeric_columns: string[];
  shape: number[];
}

export interface ClusteringParams {
  job_id: string;
  methods: string[];
  n_clusters: number;
  sigma: number;
  n_neighbors: number;
  use_pca: boolean;
  dim_reducer: string;
}

export interface ClusteringResult {
  labels: Record<string, number[]>;
  coords2D: number[][];
  coords3D: number[][];
  metrics: Record<string, Record<string, number>>;
}

export interface ProgressUpdate {
  progress: number;
  message: string;
}

export interface ApiError {
  detail: string;
  status?: number;
}

// New interfaces for Phase 2
export interface DatasetUploadResponse {
  dataset_id: string;
  stats: {
    total_rows: number;
    total_columns: number;
    file_size_kb: number;
    column_names: string[];
  };
}

export interface DatasetPreviewResponse {
  columns: string[];
  rows: any[][];
  totalRows: number;
  totalColumns: number;
  previewRows: number;
}

// API Functions

/**
 * Upload CSV data file
 * @param file - CSV file to upload
 * @returns Promise with upload response
 */
export const uploadData = async (file: File): Promise<UploadResponse> => {
  try {
    const formData = new FormData();
    formData.append('file', file);

    const response: AxiosResponse<UploadResponse> = await apiClient.post(
      '/upload',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );

    return response.data;
  } catch (error: any) {
    const apiError: ApiError = {
      detail: error.response?.data?.detail || error.message || 'Upload failed',
      status: error.response?.status,
    };
    throw apiError;
  }
};

/**
 * Run clustering algorithms
 * @param params - Clustering parameters
 * @returns Promise with clustering results
 */
export const runClustering = async (params: ClusteringParams): Promise<ClusteringResult> => {
  try {
    const response: AxiosResponse<ClusteringResult> = await apiClient.post(
      '/cluster',
      params
    );

    return response.data;
  } catch (error: any) {
    const apiError: ApiError = {
      detail: error.response?.data?.detail || error.message || 'Clustering failed',
      status: error.response?.status,
    };
    throw apiError;
  }
};

/**
 * Upload dataset file for Phase 2
 * @param file - CSV file to upload
 * @returns Promise with dataset upload response
 */
export const uploadDataset = async (file: File): Promise<DatasetUploadResponse> => {
  try {
    const formData = new FormData();
    formData.append('file', file);

    const response: AxiosResponse<DatasetUploadResponse> = await apiClient.post(
      '/upload',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );

    return response.data;
  } catch (error: any) {
    const apiError: ApiError = {
      detail: error.response?.data?.detail || error.message || 'Dataset upload failed',
      status: error.response?.status,
    };
    throw apiError;
  }
};

/**
 * Get dataset preview
 * @param datasetId - Dataset ID to preview
 * @returns Promise with dataset preview data
 */
export const getDatasetPreview = async (datasetId: string): Promise<DatasetPreviewResponse> => {
  try {
    const response: AxiosResponse<DatasetPreviewResponse> = await apiClient.get(
      `/datasets/${datasetId}/preview`
    );

    return response.data;
  } catch (error: any) {
    const apiError: ApiError = {
      detail: error.response?.data?.detail || error.message || 'Failed to load dataset preview',
      status: error.response?.status,
    };
    throw apiError;
  }
};

/**
 * Initialize progress tracking for a job
 * @param jobId - Job ID to track
 * @param callback - Function to call on progress updates
 * @returns Socket instance for manual control
 */
export const initProgress = (
  jobId: string,
  callback: (progress: ProgressUpdate) => void
): Socket => {
  const socket = io(WS_BASE_URL, {
    transports: ['websocket', 'polling'],
    autoConnect: true,
  });

  // Connect to specific job progress endpoint
  socket.on('connect', () => {
    console.log(`Connected to progress tracking for job: ${jobId}`);
    // Join the specific job room or endpoint
    socket.emit('join', jobId);
  });

  // Listen for progress updates
  socket.on('progress', (data: ProgressUpdate) => {
    console.log(`Progress update for ${jobId}:`, data);
    callback(data);
  });

  // Handle connection errors
  socket.on('connect_error', (error) => {
    console.error('WebSocket connection error:', error);
    callback({
      progress: 0,
      message: 'Connection error - retrying...',
    });
  });

  // Handle disconnection
  socket.on('disconnect', (reason) => {
    console.log('WebSocket disconnected:', reason);
    callback({
      progress: 0,
      message: 'Disconnected - attempting to reconnect...',
    });
  });

  return socket;
};

/**
 * Get available clustering methods
 * @returns Promise with list of available methods
 */
export const getAvailableMethods = async (): Promise<string[]> => {
  try {
    const response = await apiClient.get('/methods');
    return response.data.methods || [];
  } catch (error) {
    console.warn('Could not fetch available methods, using defaults');
    return ['kmeans', 'spectral', 'dbscan', 'agglomerative'];
  }
};

/**
 * Get experiment history
 * @param jobId - Optional job ID to filter by
 * @returns Promise with experiment history
 */
export const getExperimentHistory = async (jobId?: string): Promise<any[]> => {
  try {
    const params = jobId ? { job_id: jobId } : {};
    const response = await apiClient.get('/experiments', { params });
    return response.data.experiments || [];
  } catch (error) {
    console.warn('Could not fetch experiment history');
    return [];
  }
};

/**
 * Save experiment configuration
 * @param experiment - Experiment data to save
 * @returns Promise with saved experiment
 */
export const saveExperiment = async (experiment: any): Promise<any> => {
  try {
    const response = await apiClient.post('/experiments', experiment);
    return response.data;
  } catch (error: any) {
    const apiError: ApiError = {
      detail: error.response?.data?.detail || error.message || 'Save failed',
      status: error.response?.status,
    };
    throw apiError;
  }
};

/**
 * Health check endpoint
 * @returns Promise with API health status
 */
export const healthCheck = async (): Promise<{ status: string; message: string }> => {
  try {
    const response = await apiClient.get('/');
    return {
      status: 'healthy',
      message: response.data.message || 'API is running',
    };
  } catch (error) {
    return {
      status: 'unhealthy',
      message: 'API is not responding',
    };
  }
};

// Utility functions

/**
 * Create WebSocket URL for progress tracking
 * @param jobId - Job ID
 * @returns WebSocket URL
 */
export const createProgressUrl = (jobId: string): string => {
  const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = process.env.REACT_APP_WS_URL || `${wsProtocol}//${window.location.host}`;
  return `${host}/ws/progress/${jobId}`;
};

/**
 * Format API error for display
 * @param error - API error object
 * @returns Formatted error message
 */
export const formatApiError = (error: ApiError): string => {
  if (error.status) {
    return `Error ${error.status}: ${error.detail}`;
  }
  return error.detail;
};

/**
 * Check if error is network related
 * @param error - Error object
 * @returns True if network error
 */
export const isNetworkError = (error: any): boolean => {
  return !error.response && (error.code === 'NETWORK_ERROR' || error.message.includes('Network Error'));
};

/**
 * Retry API call with exponential backoff
 * @param apiCall - Function that returns a Promise
 * @param maxRetries - Maximum number of retries
 * @param baseDelay - Base delay in milliseconds
 * @returns Promise with retry logic
 */
export const retryApiCall = async <T>(
  apiCall: () => Promise<T>,
  maxRetries: number = 3,
  baseDelay: number = 1000
): Promise<T> => {
  let lastError: any;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await apiCall();
    } catch (error) {
      lastError = error;
      
      if (attempt === maxRetries || !isNetworkError(error)) {
        throw error;
      }

      const delay = baseDelay * Math.pow(2, attempt);
      console.log(`Retry attempt ${attempt + 1}/${maxRetries} after ${delay}ms`);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }

  throw lastError;
};

export default apiClient;
