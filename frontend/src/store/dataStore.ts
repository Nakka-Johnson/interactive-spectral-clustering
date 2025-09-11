import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { ClusteringRun } from '../types/api';

// Table preview interface
interface TablePreview {
  columns: string[];
  rows: any[][];
  totalRows: number;
}

/**
 * Simplified Zustand store slice based on the requested interface
 * Focuses on core data management: jobId, dataRows, columns, filters, results, progress
 */

// Type definitions for better type safety
interface RangeFilter {
  min: number;
  max: number;
}

interface FilterValue {
  [key: string]: string | number | boolean | string[] | number[] | RangeFilter | null | undefined;
}

interface DataState {
  datasetId: string | null;
  preview: TablePreview | null;
  runs: ClusteringRun[];
  activeRunId: string | null;
}

interface Store extends DataState {
  // Core data properties
  jobId: string;
  dataRows: Array<Record<string, string | number | null>>;
  columns: string[];
  filters: FilterValue;
  results: ClusteringRun | null;
  progress: number;

  // Action methods
  setJobId: (jobId: string) => void;
  setDataRows: (dataRows: Array<Record<string, string | number | null>>) => void;
  setColumns: (columns: string[]) => void;
  setFilters: (filters: FilterValue) => void;
  setResults: (results: ClusteringRun | null) => void;
  setProgress: (progress: number) => void;

  // Utility methods
  reset: () => void;
  updateFilter: (key: string, value: string | number | boolean | string[] | number[] | RangeFilter | null | undefined) => void;
  addDataRow: (row: Record<string, string | number | null>) => void;
  removeDataRow: (index: number) => void;
  getFilteredRows: () => Array<Record<string, string | number | null>>;
  isLoading: () => boolean;
  hasData: () => boolean;
}

// Initial state
const initialState = {
  jobId: '',
  dataRows: [],
  columns: [],
  filters: {},
  results: null,
  progress: 0,
  datasetId: null,
  preview: null,
  runs: [],
  activeRunId: null,
};

/**
 * Create the simplified data store
 */
export const useStore = create<Store>()(
  devtools(
    (set, get) => ({
      // Initial state
      ...initialState,

      // Core setters
      setJobId: (jobId: string) => {
        set(
          { jobId },
          false,
          'setJobId'
        );
      },

      setDataRows: (dataRows: any[]) => {
        set(
          { dataRows },
          false,
          'setDataRows'
        );
      },

      setColumns: (columns: string[]) => {
        set(
          { columns },
          false,
          'setColumns'
        );
      },

      setFilters: (filters: FilterValue) => {
        set(
          { filters },
          false,
          'setFilters'
        );
      },

      setResults: (results: any) => {
        set(
          { results },
          false,
          'setResults'
        );
      },

      setProgress: (progress: number) => {
        set(
          { progress: Math.max(0, Math.min(100, progress)) }, // Clamp between 0-100
          false,
          'setProgress'
        );
      },

      // Utility methods
      reset: () => {
        set(
          initialState,
          false,
          'reset'
        );
      },

      updateFilter: (key: string, value: any) => {
        set(
          (state) => ({
            filters: {
              ...state.filters,
              [key]: value,
            },
          }),
          false,
          'updateFilter'
        );
      },

      addDataRow: (row: any) => {
        set(
          (state) => ({
            dataRows: [...state.dataRows, row],
          }),
          false,
          'addDataRow'
        );
      },

      removeDataRow: (index: number) => {
        set(
          (state) => ({
            dataRows: state.dataRows.filter((_, i) => i !== index),
          }),
          false,
          'removeDataRow'
        );
      },

      getFilteredRows: (): any[] => {
        const { dataRows, filters } = get();
        
        if (!filters || Object.keys(filters).length === 0) {
          return dataRows;
        }

        return dataRows.filter((row) => {
          return Object.entries(filters).every(([key, filterValue]) => {
            if (filterValue === null || filterValue === undefined || filterValue === '') {
              return true; // No filter applied
            }

            const cellValue = row[key];
            
            // Handle different filter types
            if (typeof filterValue === 'string') {
              return String(cellValue).toLowerCase().includes(filterValue.toLowerCase());
            }
            
            if (typeof filterValue === 'number') {
              return Number(cellValue) === filterValue;
            }

            if (typeof filterValue === 'boolean') {
              return Boolean(cellValue) === filterValue;
            }

            if (Array.isArray(filterValue)) {
              return cellValue !== null && (filterValue as (string | number)[]).includes(cellValue);
            }

            if (typeof filterValue === 'object' && filterValue !== null && 'min' in filterValue && 'max' in filterValue) {
              const rangeFilter = filterValue as RangeFilter;
              const numValue = Number(cellValue);
              return numValue >= rangeFilter.min && numValue <= rangeFilter.max;
            }

            return String(cellValue) === String(filterValue);
          });
        });
      },

      isLoading: (): boolean => {
        const { progress } = get();
        return progress > 0 && progress < 100;
      },

      hasData: (): boolean => {
        const { dataRows, columns } = get();
        return dataRows.length > 0 && columns.length > 0;
      },
    }),
    {
      name: 'data-store', // Name for devtools
    }
  )
);

/**
 * Utility hooks for specific parts of the store
 */
export const useJobId = () => useStore((state) => state.jobId);
export const useDataRows = () => useStore((state) => state.dataRows);
export const useColumns = () => useStore((state) => state.columns);
export const useFilters = () => useStore((state) => state.filters);
export const useResults = () => useStore((state) => state.results);
export const useProgress = () => useStore((state) => state.progress);

/**
 * Computed value hooks
 */
export const useFilteredData = () => useStore((state) => state.getFilteredRows());
export const useIsLoading = () => useStore((state) => state.isLoading());
export const useHasData = () => useStore((state) => state.hasData());

/**
 * Action hooks (for performance optimization)
 */
export const useDataActions = () => useStore((state) => ({
  setJobId: state.setJobId,
  setDataRows: state.setDataRows,
  setColumns: state.setColumns,
  setFilters: state.setFilters,
  setResults: state.setResults,
  setProgress: state.setProgress,
  reset: state.reset,
  updateFilter: state.updateFilter,
  addDataRow: state.addDataRow,
  removeDataRow: state.removeDataRow,
}));

/**
 * Example usage:
 * 
 * // Basic usage
 * const { jobId, dataRows, setJobId, setDataRows } = useStore();
 * 
 * // Performance optimized usage
 * const jobId = useJobId();
 * const { setJobId, setDataRows } = useDataActions();
 * 
 * // Computed values
 * const filteredData = useFilteredData();
 * const isLoading = useIsLoading();
 * 
 * // Update data
 * setJobId('job_123');
 * setDataRows([{ id: 1, name: 'John' }, { id: 2, name: 'Jane' }]);
 * setColumns(['id', 'name']);
 * setFilters({ name: 'John' });
 * setProgress(50);
 */

export default useStore;
