/**
 * Zustand store for Interactive Spectral Clustering Platform.
 * 
 * Manages global application state including datasets, clustering runs,
 * and active selections. Provides actions for fetching and updating state.
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { apiClient } from '../lib/http';
import type {
  ClusteringRun,
  UploadResponse,
  AlgorithmId,
  RunStatus,
  ParameterMap
} from '../types/api';

// Helper for shallow equality comparison
const shallowEqualObj = (a: any, b: any) => {
  if (Object.is(a, b)) return true;
  if (!a || !b || typeof a !== 'object' || typeof b !== 'object') return false;
  const ka = Object.keys(a), kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  for (const k of ka) if (!Object.is((a as any)[k], (b as any)[k])) return false;
  return true;
};

/**
 * Interface for uploaded dataset with metadata.
 */
export interface Dataset {
  job_id: string;
  name: string;
  columns: string[];
  numeric_columns: string[];
  shape: [number, number];
  upload_time: string;
  file_name?: string;
  file_size?: number;
}

/**
 * Interface for clustering run with additional UI state.
 */
export interface ClusteringRunState {
  // Core fields from backend
  run_id: string;
  dataset_job_id?: string;
  algorithm: AlgorithmId;
  parameters: ParameterMap;
  status: RunStatus;
  started_at?: string;
  ended_at?: string;
  progress?: number;
  labels?: number[];
  cluster_centers?: number[][];
  metrics?: any;
  embedding_2d?: number[][];
  embedding_3d?: number[][];
  execution_time_seconds?: number;
  gpu_used?: boolean;
  memory_usage_mb?: number;
  error_message?: string;
  logs?: string[];
  expires_at?: string;
  
  // UI state
  dataset_name?: string;
  is_loading?: boolean;
  
  // Computed properties for UI compatibility
  id: string;          // maps to run_id
  dataset: string;     // maps to dataset_job_id or dataset_name
  createdAt: string;   // maps to started_at
  duration?: number;   // maps to execution_time_seconds
  results?: {
    embedding?: number[][];
    labels?: number[];
    centers?: number[][];
    metrics?: Record<string, number>;
  };
}

/**
 * Application state interface.
 */
interface AppState {
  // Data state
  datasets: Dataset[];
  runs: ClusteringRunState[];
  
  // Active selections
  activeDataset: Dataset | null;
  activeRunId: string | null;
  selectedRunId: string | null;
  
  // UI state
  isLoading: boolean;
  error: string | null;
  
  // Upload state
  uploadProgress: number;
  isUploading: boolean;
}

/**
 * Application actions interface.
 */
interface AppActions {
  // Dataset management
  fetchDatasets: () => Promise<void>;
  uploadDataset: (file: File, name?: string) => Promise<UploadResponse>;
  setActiveDataset: (dataset: Dataset | null) => void;
  
  // Clustering runs management
  fetchRuns: (datasetJobId?: string) => Promise<void>;
  refreshRuns: () => Promise<void>;
  addRun: (run: ClusteringRunState) => void;
  updateRun: (runId: string, updates: Partial<ClusteringRunState>) => void;
  setActiveRun: (runId: string | null) => void;
  setSelectedRun: (runId: string | null) => void;
  
  // UI state management
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  clearError: () => void;
  
  // Utility actions
  reset: () => void;
}

/**
 * Combined store interface.
 */
type AppStore = AppState & AppActions;

/**
 * Initial state values.
 */
const initialState: AppState = {
  datasets: [],
  runs: [],
  activeDataset: null,
  activeRunId: null,
  selectedRunId: null,
  isLoading: false,
  error: null,
  uploadProgress: 0,
  isUploading: false,
};

/**
 * Create the Zustand store with persistence and devtools.
 */
export const useAppStore = create<AppStore>()(
  devtools(
    persist(
      (set, get) => ({
        ...initialState,

        // Dataset management
        fetchDatasets: async () => {
          try {
            set({ isLoading: true, error: null });
            
            // Note: This endpoint doesn't exist yet, so we'll mock it for now
            // const response = await apiClient.httpClient.get('/datasets');
            // set({ datasets: response.data, isLoading: false });
            
            // Mock datasets for now
            const mockDatasets: Dataset[] = [
              {
                job_id: 'dataset_001',
                name: 'Iris Dataset',
                columns: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'],
                numeric_columns: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
                shape: [150, 4],
                upload_time: new Date().toISOString(),
                file_name: 'iris.csv',
                file_size: 4096
              },
              {
                job_id: 'dataset_002', 
                name: 'Wine Dataset',
                columns: ['alcohol', 'malic_acid', 'ash', 'alcalinity', 'magnesium', 'total_phenols'],
                numeric_columns: ['alcohol', 'malic_acid', 'ash', 'alcalinity', 'magnesium', 'total_phenols'],
                shape: [178, 6],
                upload_time: new Date(Date.now() - 86400000).toISOString(),
                file_name: 'wine.csv',
                file_size: 8192
              }
            ];
            
            set((state) => {
              if (shallowEqualObj(state.datasets, mockDatasets)) return state;
              return { ...state, datasets: mockDatasets, isLoading: false };
            });
          } catch (error) {
            console.error('Failed to fetch datasets:', error);
            set({ 
              error: error instanceof Error ? error.message : 'Failed to fetch datasets',
              isLoading: false 
            });
          }
        },

        uploadDataset: async (file: File, name?: string) => {
          try {
            set({ isUploading: true, uploadProgress: 0, error: null });
            
            // Simulate upload progress
            const progressInterval = setInterval(() => {
              set((state) => ({
                uploadProgress: Math.min(state.uploadProgress + 20, 90)
              }));
            }, 200);
            
            const response = await apiClient.uploadFile(file);
            
            clearInterval(progressInterval);
            set({ uploadProgress: 100 });
            
            // Create dataset from upload response
            const newDataset: Dataset = {
              job_id: response.job_id,
              name: name || file.name.replace(/\.[^/.]+$/, ''),
              columns: response.columns,
              numeric_columns: response.numeric_columns,
              shape: response.shape,
              upload_time: new Date().toISOString(),
              file_name: file.name,
              file_size: file.size
            };
            
            // Add to datasets
            set((state) => {
              const nextDatasets = [...state.datasets, newDataset];
              if (shallowEqualObj(state.datasets, nextDatasets)) return state;
              return {
                ...state,
                datasets: nextDatasets,
                isUploading: false,
                uploadProgress: 0
              };
            });
            
            return response;
          } catch (error) {
            console.error('Failed to upload dataset:', error);
            set({ 
              error: error instanceof Error ? error.message : 'Failed to upload dataset',
              isUploading: false,
              uploadProgress: 0
            });
            throw error;
          }
        },

        setActiveDataset: (dataset: Dataset | null) => {
          set({ activeDataset: dataset });
          
          // Auto-fetch runs for the selected dataset
          if (dataset) {
            get().fetchRuns(dataset.job_id);
          }
        },

        // Clustering runs management
        fetchRuns: async (datasetJobId?: string) => {
          try {
            set({ isLoading: true, error: null });
            
            // Mock runs for now since backend endpoints don't exist yet
            const mockRuns: ClusteringRunState[] = [
              {
                run_id: 'run_001',
                dataset_job_id: 'dataset_001',
                algorithm: 'spectral' as AlgorithmId,
                parameters: { n_clusters: 3, sigma: 1.0 } as ParameterMap,
                status: 'completed' as RunStatus,
                started_at: new Date(Date.now() - 3600000).toISOString(),
                ended_at: new Date(Date.now() - 3500000).toISOString(),
                labels: [0, 0, 1, 1, 2, 2],
                metrics: {
                  silhouette_score: 0.75,
                  davies_bouldin_score: 0.65,
                  calinski_harabasz_score: 561.63,
                  inertia: 78.85
                },
                execution_time_seconds: 100,
                gpu_used: true,
                dataset_name: 'Iris Dataset',
                // Computed fields for UI compatibility
                id: 'run_001',
                dataset: 'Iris Dataset',
                createdAt: new Date(Date.now() - 3600000).toISOString(),
                duration: 100,
                results: {
                  labels: [0, 0, 1, 1, 2, 2],
                  metrics: {
                    silhouette_score: 0.75,
                    davies_bouldin_score: 0.65,
                    calinski_harabasz_score: 561.63,
                    inertia: 78.85
                  }
                }
              },
              {
                run_id: 'run_002',
                dataset_job_id: 'dataset_001',
                algorithm: 'kmeans' as AlgorithmId,
                parameters: { n_clusters: 4, init: 'k-means++' },
                status: 'running' as RunStatus,
                started_at: new Date(Date.now() - 300000).toISOString(),
                dataset_name: 'Iris Dataset',
                is_loading: true,
                // Computed fields for UI compatibility
                id: 'run_002',
                dataset: 'Iris Dataset',
                createdAt: new Date(Date.now() - 300000).toISOString()
              },
              {
                run_id: 'run_003',
                dataset_job_id: 'dataset_002',
                algorithm: 'dbscan' as AlgorithmId,
                parameters: { eps: 0.5, min_samples: 5 },
                status: 'error' as RunStatus,
                started_at: new Date(Date.now() - 1800000).toISOString(),
                ended_at: new Date(Date.now() - 1750000).toISOString(),
                error_message: 'Insufficient memory for computation',
                dataset_name: 'Wine Dataset',
                // Computed fields for UI compatibility
                id: 'run_003',
                dataset: 'Wine Dataset',
                createdAt: new Date(Date.now() - 1800000).toISOString(),
                duration: 50
              }
            ];
            
            // Filter by dataset if specified
            const filteredRuns = datasetJobId 
              ? mockRuns.filter(run => run.dataset_job_id === datasetJobId)
              : mockRuns;
            
            set((state) => {
              if (shallowEqualObj(state.runs, filteredRuns)) return state;
              return { ...state, runs: filteredRuns, isLoading: false };
            });
          } catch (error) {
            console.error('Failed to fetch runs:', error);
            set({ 
              error: error instanceof Error ? error.message : 'Failed to fetch runs',
              isLoading: false 
            });
          }
        },

        addRun: (run: ClusteringRunState) => {
          set((state) => {
            const nextRuns = [run, ...state.runs];
            if (shallowEqualObj(state.runs, nextRuns)) return state;
            return { ...state, runs: nextRuns };
          });
        },

        updateRun: (runId: string, updates: Partial<ClusteringRunState>) => {
          set((state) => {
            const nextRuns = state.runs.map(run =>
              run.run_id === runId ? { ...run, ...updates } : run
            );
            if (shallowEqualObj(state.runs, nextRuns)) return state;
            return { ...state, runs: nextRuns };
          });
        },

        setActiveRun: (runId: string | null) => {
          set({ activeRunId: runId });
        },

        refreshRuns: async () => {
          return get().fetchRuns();
        },

        setSelectedRun: (runId: string | null) => {
          set({ selectedRunId: runId });
        },

        // UI state management
        setLoading: (loading: boolean) => {
          set({ isLoading: loading });
        },

        setError: (error: string | null) => {
          set({ error });
        },

        clearError: () => {
          set({ error: null });
        },

        // Utility actions
        reset: () => {
          set(initialState);
        },
      }),
      {
        name: 'clustering-app-store',
        partialize: (state) => ({
          // Only persist certain parts of the state
          datasets: state.datasets,
          activeDataset: state.activeDataset,
          activeRunId: state.activeRunId,
          selectedRunId: state.selectedRunId,
        }),
      }
    ),
    {
      name: 'clustering-app-store',
    }
  )
);

/**
 * Selector hooks for easier access to specific parts of the store.
 * Using direct selectors to prevent unnecessary re-renders.
 */
export const useDatasets = () => useAppStore((state) => state.datasets);
export const useRuns = () => useAppStore((state) => state.runs);
export const useActiveDataset = () => useAppStore((state) => state.activeDataset);
export const useActiveRunId = () => useAppStore((state) => state.activeRunId);
export const useSelectedRunId = () => useAppStore((state) => state.selectedRunId);
export const useActiveRun = () => {
  const activeRunId = useAppStore((state) => state.activeRunId);
  const runs = useAppStore((state) => state.runs);
  if (!activeRunId) return null;
  return runs.find(run => run.run_id === activeRunId) || null;
};
export const useSelectedRun = () => {
  const selectedRunId = useAppStore((state) => state.selectedRunId);
  const runs = useAppStore((state) => state.runs);
  if (!selectedRunId) return null;
  return runs.find(run => run.run_id === selectedRunId) || null;
};
export const useLoading = () => useAppStore((state) => state.isLoading);
export const useError = () => useAppStore((state) => state.error);
export const useIsUploading = () => useAppStore((state) => state.isUploading);
export const useUploadProgress = () => useAppStore((state) => state.uploadProgress);

// Composite selector that uses shallow comparison
export const useUploadState = () => {
  const isUploading = useAppStore((state) => state.isUploading);
  const uploadProgress = useAppStore((state) => state.uploadProgress);
  return { isUploading, uploadProgress };
};
