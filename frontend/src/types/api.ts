/**
 * TypeScript type definitions for Interactive Spectral Clustering Platform API.
 * 
 * These types mirror the Pydantic models from the backend to ensure type safety
 * across the full stack. All API requests and responses should use these types.
 */

/**
 * Supported clustering algorithms.
 * Each algorithm has specific parameter requirements documented in the backend.
 */
export type AlgorithmId = "spectral" | "kmeans" | "dbscan" | "gmm" | "agglomerative";

/**
 * Dimensionality reduction methods for visualization.
 */
export type EmbeddingMethod = "pca" | "tsne" | "umap";

/**
 * Clustering run execution status for progress tracking.
 */
export type RunStatus = "queued" | "running" | "completed" | "done" | "error";

/**
 * Reference to an uploaded dataset with metadata.
 */
export interface DatasetRef {
  job_id: string;
  name?: string;
  columns: string[];
  numeric_columns: string[];
  shape: [number, number]; // [rows, columns]
  upload_time: string; // ISO datetime string
}

/**
 * Algorithm-specific parameters as a flexible key-value store.
 * Different algorithms require different parameters.
 */
export interface ParameterMap {
  // Common parameters
  n_clusters?: number; // 2-50
  random_state?: number;
  use_gpu?: boolean;
  
  // Spectral clustering specific
  sigma?: number; // > 0
  n_neighbors?: number; // >= 3
  
  // K-means specific
  init?: string; // "k-means++", "random"
  max_iter?: number; // >= 1
  
  // DBSCAN specific
  eps?: number; // > 0
  min_samples?: number; // >= 1
  
  // GMM specific
  n_components?: number; // >= 1
  covariance_type?: string; // "full", "tied", "diag", "spherical"
  
  // Agglomerative specific
  linkage?: string; // "ward", "complete", "average", "single"
  metric?: string; // "euclidean", "manhattan", etc.
  
  // Additional parameters
  extra_params?: Record<string, any>;
}

/**
 * Request to perform clustering analysis on a dataset.
 */
export interface ClusteringRequest {
  algorithm: AlgorithmId;
  parameters?: ParameterMap;
  
  // Data source - exactly one must be provided
  dataset_ref?: DatasetRef;
  inline_data?: number[][]; // 2D array of numeric data
  
  // Execution options
  seed?: number;
  selected_columns?: string[];
  preprocessing?: Record<string, any>;
}

/**
 * Clustering evaluation metrics and performance indicators.
 */
export interface ClusteringMetrics {
  silhouette_score?: number; // -1 to 1
  davies_bouldin_score?: number; // lower is better
  calinski_harabasz_score?: number; // higher is better  
  adjusted_rand_score?: number; // 0 to 1
  inertia?: number; // sum of squared distances to centroids
  n_clusters_found?: number; // number of clusters identified
  
  // Custom metrics
  custom_metrics?: Record<string, number>;
}

/**
 * Complete clustering run with results, metrics, and execution metadata.
 */
export interface ClusteringRun {
  // Identification
  run_id: string;
  dataset_job_id?: string;
  
  // Request details
  algorithm: AlgorithmId;
  parameters: ParameterMap;
  
  // Execution tracking
  status: RunStatus;
  started_at?: string; // ISO datetime
  ended_at?: string; // ISO datetime
  progress?: number; // 0-100 progress percentage
  
  // Results
  labels?: number[]; // cluster assignments
  cluster_centers?: number[][]; // centroids if applicable
  metrics?: ClusteringMetrics;
  
  // Visualization data
  embedding_2d?: number[][]; // 2D coordinates
  embedding_3d?: number[][]; // 3D coordinates
  
  // Execution metadata  
  execution_time_seconds?: number;
  gpu_used?: boolean;
  memory_usage_mb?: number;
  
  // Error handling
  error_message?: string;
  logs?: string[];
  
  // Cleanup
  expires_at?: string; // ISO datetime
}

/**
 * Request for hyperparameter grid search.
 */
export interface GridSearchRequest {
  algorithm: AlgorithmId;
  param_grid: Record<string, any[]>; // parameter combinations
  dataset_ref: DatasetRef;
  
  // Search options
  cv_folds?: number; // >= 2
  scoring_metric?: string;
  n_jobs?: number; // -1 for all cores
  seed?: number;
  
  // Resource limits
  max_combinations?: number;
  timeout_minutes?: number;
}

/**
 * Results from hyperparameter grid search.
 */
export interface GridSearchResult {
  search_id: string;
  best_parameters: ParameterMap;
  best_score: number;
  all_results: Array<Record<string, any>>;
  search_time_seconds: number;
}

/**
 * Request for dimensionality reduction and embedding generation.
 */
export interface EmbeddingRequest {
  method: EmbeddingMethod;
  
  // Data source - exactly one must be provided
  dataset_ref?: DatasetRef;
  clustering_run_id?: string;
  
  // Embedding parameters
  n_components: 2 | 3; // 2D or 3D
  params?: Record<string, any>; // method-specific parameters
  
  // Data preprocessing
  use_standardization?: boolean;
  selected_features?: string[];
}

/**
 * Results from dimensionality reduction embedding.
 */
export interface EmbeddingResult {
  embedding_id: string;
  method: EmbeddingMethod;
  coordinates: number[][];
  explained_variance_ratio?: number[]; // PCA only
  processing_time_seconds: number;
}

// API Response types
export interface ClusteringResponse {
  run: ClusteringRun;
  message: string;
}

export interface GridSearchResponse {
  result: GridSearchResult;
  message: string;
}

export interface EmbeddingResponse {
  result: EmbeddingResult;
  message: string;
}

// JWT Authentication types (from existing implementation)
export interface LoginRequest {
  username: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export interface User {
  username: string;
  email?: string;
  disabled?: boolean;
}

// File upload types (from existing implementation) 
export interface UploadResponse {
  job_id: string;
  columns: string[];
  numeric_columns: string[];
  shape: [number, number];
}

// Error response type
export interface ErrorResponse {
  error: string;
  details?: any;
}

// WebSocket message types for real-time updates
export interface ProgressUpdate {
  progress: number; // 0-100
  message: string;
  run_id?: string;
}

export interface WebSocketMessage {
  type: 'progress' | 'completion' | 'error';
  data: ProgressUpdate | ClusteringRun | ErrorResponse;
}

// PHASE 0 Required Types - Aliases for compatibility
export type ClusteringMethod = 'kmeans' | 'spectral' | 'dbscan';
