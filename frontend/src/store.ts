import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { Socket } from 'socket.io-client';
import { ClusteringParams, ClusteringResult, ProgressUpdate } from './api';

// Type definitions for the store
export interface DataFilter {
  columnFilters?: string[];
  rowConditions?: Record<string, any>;
  useAllColumns?: boolean;
}

export interface ClusteringParameters {
  methods: string[];
  n_clusters: number;
  sigma: number;
  n_neighbors: number;
  use_pca: boolean;
  dim_reducer: 'pca' | 'tsne' | 'umap';
  visualization_method?: 'pca' | 'tsne' | 'umap';
}

export interface ExperimentResult {
  id: string;
  timestamp: Date;
  jobId: string;
  parameters: ClusteringParameters;
  filters: DataFilter;
  result: ClusteringResult;
  metrics: Record<string, Record<string, number>>;
  executionTime?: number;
  notes?: string;
}

export interface Dataset {
  jobId: string;
  filename: string;
  columns: string[];
  numericColumns: string[];
  shape: number[];
  uploadTime: Date;
}

export interface ProgressState {
  isRunning: boolean;
  progress: number;
  message: string;
  socket?: Socket;
}

export interface UIState {
  currentTab: 'upload' | 'explore' | 'configure' | 'visualize' | 'metrics' | 'results' | 'history' | 'report';
  selectedExperiments: string[];
  comparisonMode: boolean;
  showAdvancedOptions: boolean;
  darkMode: boolean;
}

// Main store interface
export interface ClusteringStore {
  // Dataset state
  dataset: Dataset | null;
  
  // Filters and parameters
  filters: DataFilter;
  parameters: ClusteringParameters;
  
  // Results and progress
  currentResult: ClusteringResult | null;
  progress: ProgressState;
  
  // Experiment history
  experiments: ExperimentResult[];
  
  // UI state
  ui: UIState;
  
  // Error handling
  error: string | null;
  
  // Actions - Dataset
  setDataset: (dataset: Dataset) => void;
  clearDataset: () => void;
  
  // Actions - Filters
  setFilters: (filters: Partial<DataFilter>) => void;
  resetFilters: () => void;
  
  // Actions - Parameters
  setParameters: (params: Partial<ClusteringParameters>) => void;
  resetParameters: () => void;
  
  // Actions - Results
  setResult: (result: ClusteringResult) => void;
  clearResult: () => void;
  
  // Actions - Progress
  setProgress: (progress: Partial<ProgressState>) => void;
  updateProgress: (update: ProgressUpdate) => void;
  resetProgress: () => void;
  
  // Actions - Experiments
  addExperiment: (experiment: Omit<ExperimentResult, 'id' | 'timestamp'>) => void;
  removeExperiment: (id: string) => void;
  loadExperiment: (id: string) => void;
  clearExperiments: () => void;
  
  // Actions - UI
  setUI: (ui: Partial<UIState>) => void;
  setCurrentTab: (tab: UIState['currentTab']) => void;
  toggleComparisonMode: () => void;
  toggleDarkMode: () => void;
  
  // Actions - Error handling
  setError: (error: string | null) => void;
  clearError: () => void;
  
  // Actions - Complex operations
  prepareClusteringParams: () => ClusteringParams;
  getSelectedExperiments: () => ExperimentResult[];
  getExperimentById: (id: string) => ExperimentResult | undefined;
  
  // Actions - Validation
  validateParameters: () => { isValid: boolean; errors: string[] };
  canRunClustering: () => boolean;
}

// Default values
const defaultFilters: DataFilter = {
  columnFilters: undefined,
  rowConditions: {},
  useAllColumns: true,
};

const defaultParameters: ClusteringParameters = {
  methods: ['kmeans', 'spectral'],
  n_clusters: 3,
  sigma: 1.0,
  n_neighbors: 10,
  use_pca: false,
  dim_reducer: 'pca',
  visualization_method: 'pca',
};

const defaultProgress: ProgressState = {
  isRunning: false,
  progress: 0,
  message: '',
  socket: undefined,
};

const defaultUI: UIState = {
  currentTab: 'upload',
  selectedExperiments: [],
  comparisonMode: false,
  showAdvancedOptions: false,
  darkMode: false,
};

// Create the store
export const useClusteringStore = create<ClusteringStore>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial state
        dataset: null,
        filters: defaultFilters,
        parameters: defaultParameters,
        currentResult: null,
        progress: defaultProgress,
        experiments: [],
        ui: defaultUI,
        error: null,

        // Dataset actions
        setDataset: (dataset: Dataset) => {
          set(
            (state) => ({
              dataset,
              error: null,
              // Reset filters when new dataset is loaded
              filters: {
                ...defaultFilters,
                columnFilters: dataset.numericColumns,
              },
            }),
            false,
            'setDataset'
          );
        },

        clearDataset: () => {
          set(
            (state) => ({
              dataset: null,
              currentResult: null,
              filters: defaultFilters,
              error: null,
            }),
            false,
            'clearDataset'
          );
        },

        // Filter actions
        setFilters: (filters: Partial<DataFilter>) => {
          set(
            (state) => ({
              filters: { ...state.filters, ...filters },
            }),
            false,
            'setFilters'
          );
        },

        resetFilters: () => {
          set(
            (state) => ({
              filters: defaultFilters,
            }),
            false,
            'resetFilters'
          );
        },

        // Parameter actions
        setParameters: (params: Partial<ClusteringParameters>) => {
          set(
            (state) => ({
              parameters: { ...state.parameters, ...params },
            }),
            false,
            'setParameters'
          );
        },

        resetParameters: () => {
          set(
            (state) => ({
              parameters: defaultParameters,
            }),
            false,
            'resetParameters'
          );
        },

        // Result actions
        setResult: (result: ClusteringResult) => {
          set(
            (state) => ({
              currentResult: result,
              error: null,
            }),
            false,
            'setResult'
          );
        },

        clearResult: () => {
          set(
            (state) => ({
              currentResult: null,
            }),
            false,
            'clearResult'
          );
        },

        // Progress actions
        setProgress: (progress: Partial<ProgressState>) => {
          set(
            (state) => ({
              progress: { ...state.progress, ...progress },
            }),
            false,
            'setProgress'
          );
        },

        updateProgress: (update: ProgressUpdate) => {
          set(
            (state) => ({
              progress: {
                ...state.progress,
                progress: update.progress,
                message: update.message,
                isRunning: update.progress < 100 && update.progress > 0,
              },
            }),
            false,
            'updateProgress'
          );
        },

        resetProgress: () => {
          set(
            (state) => ({
              progress: defaultProgress,
            }),
            false,
            'resetProgress'
          );
        },

        // Experiment actions
        addExperiment: (experiment: Omit<ExperimentResult, 'id' | 'timestamp'>) => {
          const newExperiment: ExperimentResult = {
            ...experiment,
            id: `exp_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            timestamp: new Date(),
          };

          set(
            (state) => ({
              experiments: [newExperiment, ...state.experiments],
            }),
            false,
            'addExperiment'
          );
        },

        removeExperiment: (id: string) => {
          set(
            (state) => ({
              experiments: state.experiments.filter((exp) => exp.id !== id),
              ui: {
                ...state.ui,
                selectedExperiments: state.ui.selectedExperiments.filter((expId) => expId !== id),
              },
            }),
            false,
            'removeExperiment'
          );
        },

        loadExperiment: (id: string) => {
          const experiment = get().experiments.find((exp) => exp.id === id);
          if (experiment) {
            set(
              (state) => ({
                parameters: experiment.parameters,
                filters: experiment.filters,
                currentResult: experiment.result,
              }),
              false,
              'loadExperiment'
            );
          }
        },

        clearExperiments: () => {
          set(
            (state) => ({
              experiments: [],
              ui: {
                ...state.ui,
                selectedExperiments: [],
                comparisonMode: false,
              },
            }),
            false,
            'clearExperiments'
          );
        },

        // UI actions
        setUI: (ui: Partial<UIState>) => {
          set(
            (state) => ({
              ui: { ...state.ui, ...ui },
            }),
            false,
            'setUI'
          );
        },

        setCurrentTab: (tab: UIState['currentTab']) => {
          set(
            (state) => ({
              ui: { ...state.ui, currentTab: tab },
            }),
            false,
            'setCurrentTab'
          );
        },

        toggleComparisonMode: () => {
          set(
            (state) => ({
              ui: {
                ...state.ui,
                comparisonMode: !state.ui.comparisonMode,
                selectedExperiments: state.ui.comparisonMode ? [] : state.ui.selectedExperiments,
              },
            }),
            false,
            'toggleComparisonMode'
          );
        },

        toggleDarkMode: () => {
          set(
            (state) => ({
              ui: { ...state.ui, darkMode: !state.ui.darkMode },
            }),
            false,
            'toggleDarkMode'
          );
        },

        // Error actions
        setError: (error: string | null) => {
          set(
            (state) => ({
              error,
            }),
            false,
            'setError'
          );
        },

        clearError: () => {
          set(
            (state) => ({
              error: null,
            }),
            false,
            'clearError'
          );
        },

        // Complex operations
        prepareClusteringParams: (): ClusteringParams => {
          const state = get();
          if (!state.dataset) {
            throw new Error('No dataset loaded');
          }

          return {
            job_id: state.dataset.jobId,
            methods: state.parameters.methods,
            n_clusters: state.parameters.n_clusters,
            sigma: state.parameters.sigma,
            n_neighbors: state.parameters.n_neighbors,
            use_pca: state.parameters.use_pca,
            dim_reducer: state.parameters.dim_reducer,
          };
        },

        getSelectedExperiments: (): ExperimentResult[] => {
          const state = get();
          return state.experiments.filter((exp) => 
            state.ui.selectedExperiments.includes(exp.id)
          );
        },

        getExperimentById: (id: string): ExperimentResult | undefined => {
          const state = get();
          return state.experiments.find((exp) => exp.id === id);
        },

        // Validation
        validateParameters: (): { isValid: boolean; errors: string[] } => {
          const state = get();
          const errors: string[] = [];

          if (state.parameters.methods.length === 0) {
            errors.push('At least one clustering method must be selected');
          }

          if (state.parameters.n_clusters < 1) {
            errors.push('Number of clusters must be at least 1');
          }

          if (state.parameters.sigma <= 0) {
            errors.push('Sigma must be positive');
          }

          if (state.parameters.n_neighbors < 1) {
            errors.push('Number of neighbors must be at least 1');
          }

          if (!state.dataset) {
            errors.push('No dataset loaded');
          }

          return {
            isValid: errors.length === 0,
            errors,
          };
        },

        canRunClustering: (): boolean => {
          const state = get();
          const validation = get().validateParameters();
          return validation.isValid && !state.progress.isRunning;
        },
      }),
      {
        name: 'clustering-store',
        // Only persist certain parts of the state
        partialize: (state) => ({
          experiments: state.experiments,
          ui: {
            darkMode: state.ui.darkMode,
            showAdvancedOptions: state.ui.showAdvancedOptions,
          },
          parameters: state.parameters,
        }),
      }
    ),
    {
      name: 'clustering-store',
    }
  )
);

// Utility hooks for specific parts of the store
export const useDataset = () => useClusteringStore((state) => state.dataset);
export const useParameters = () => useClusteringStore((state) => state.parameters);
export const useFilters = () => useClusteringStore((state) => state.filters);
export const useResult = () => useClusteringStore((state) => state.currentResult);
export const useProgress = () => useClusteringStore((state) => state.progress);
export const useExperiments = () => useClusteringStore((state) => state.experiments);
export const useUI = () => useClusteringStore((state) => state.ui);
export const useError = () => useClusteringStore((state) => state.error);

// Utility function to reset entire store
export const resetStore = () => {
  useClusteringStore.setState({
    dataset: null,
    filters: defaultFilters,
    parameters: defaultParameters,
    currentResult: null,
    progress: defaultProgress,
    experiments: [],
    ui: defaultUI,
    error: null,
  });
};
