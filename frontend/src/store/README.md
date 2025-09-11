# Zustand Data Store Slice

A simplified Zustand store implementation focusing on core data management for the Interactive Spectral Clustering Platform.

## 📁 File Structure

```
frontend/src/store/
├── dataStore.ts           # Main store implementation
└── store.ts              # Original comprehensive store (existing)

frontend/src/components/
└── DataStoreExample.tsx   # Usage example component
```

## 🏗️ Store Interface

The store implements the exact interface you requested:

```typescript
interface Store {
  // Core data properties
  jobId: string;
  dataRows: any[];
  columns: string[];
  filters: any;
  results: any;
  progress: number;

  // Action methods
  setJobId: (jobId: string) => void;
  setDataRows: (dataRows: any[]) => void;
  setColumns: (columns: string[]) => void;
  setFilters: (filters: any) => void;
  setResults: (results: any) => void;
  setProgress: (progress: number) => void;
}
```

## 🚀 Usage Examples

### Basic Usage
```typescript
import { useStore } from './store/dataStore';

const MyComponent = () => {
  const { 
    jobId, 
    dataRows, 
    setJobId, 
    setDataRows 
  } = useStore();

  return (
    <div>
      <p>Job ID: {jobId}</p>
      <p>Rows: {dataRows.length}</p>
      <button onClick={() => setJobId('new-job-123')}>
        Set Job ID
      </button>
    </div>
  );
};
```

### Performance Optimized Usage
```typescript
import { 
  useJobId, 
  useDataRows, 
  useDataActions 
} from './store/dataStore';

const OptimizedComponent = () => {
  // Only re-renders when jobId changes
  const jobId = useJobId();
  
  // Only re-renders when dataRows changes
  const dataRows = useDataRows();
  
  // Actions don't cause re-renders
  const { setJobId, setDataRows } = useDataActions();

  return (
    <div>
      <p>Job ID: {jobId}</p>
      <p>Rows: {dataRows.length}</p>
    </div>
  );
};
```

### Computed Values
```typescript
import { 
  useFilteredData, 
  useIsLoading, 
  useHasData 
} from './store/dataStore';

const DataDisplay = () => {
  const filteredData = useFilteredData(); // Automatically filtered data
  const isLoading = useIsLoading();       // true when 0 < progress < 100
  const hasData = useHasData();          // true when data exists

  if (isLoading) return <div>Loading...</div>;
  if (!hasData) return <div>No data available</div>;

  return (
    <div>
      <h3>Filtered Data ({filteredData.length} rows)</h3>
      {/* Render filtered data */}
    </div>
  );
};
```

## 🔧 Available Hooks

### State Hooks
- `useJobId()` - Returns current job ID
- `useDataRows()` - Returns all data rows
- `useColumns()` - Returns column definitions
- `useFilters()` - Returns active filters
- `useResults()` - Returns analysis results
- `useProgress()` - Returns current progress (0-100)

### Computed Hooks
- `useFilteredData()` - Returns filtered data based on current filters
- `useIsLoading()` - Returns true when progress is between 0 and 100
- `useHasData()` - Returns true when data exists

### Action Hooks
- `useDataActions()` - Returns all action methods without state

## 🎯 Features

### ✅ Type Safety
- Proper TypeScript interfaces
- Type-safe filter definitions
- Range filter support for numeric values

### ✅ Performance Optimized
- Individual hooks prevent unnecessary re-renders
- Computed values are cached
- Actions separated from state

### ✅ Advanced Filtering
```typescript
// String filter (case-insensitive contains)
setFilters({ name: 'john' });

// Exact value filter
setFilters({ department: 'Engineering' });

// Array filter (includes)
setFilters({ status: ['active', 'pending'] });

// Range filter for numbers
setFilters({ age: { min: 25, max: 35 } });

// Multiple filters
setFilters({ 
  department: 'Engineering',
  age: { min: 25, max: 35 },
  name: 'john'
});
```

### ✅ Utility Methods
```typescript
const { 
  reset,           // Reset entire store
  updateFilter,    // Update single filter
  addDataRow,      // Add new row
  removeDataRow,   // Remove row by index
  getFilteredRows, // Get filtered data
  isLoading,       // Check loading state
  hasData         // Check if data exists
} = useStore();

// Examples
updateFilter('department', 'Engineering');
addDataRow({ id: 5, name: 'New Person' });
removeDataRow(2); // Remove 3rd row
```

### ✅ DevTools Integration
- Redux DevTools support for debugging
- Action names for easy tracking
- State inspection and time travel

## 🔄 Integration with Existing Store

This simplified store can work alongside the existing comprehensive store:

```typescript
// Use both stores in the same component
import { useClusteringStore } from './store'; // Original store
import { useStore as useDataStore } from './store/dataStore'; // New store

const MyComponent = () => {
  // Original comprehensive store
  const { dataset, parameters } = useClusteringStore();
  
  // New simplified store
  const { dataRows, setDataRows } = useDataStore();
  
  // Use both as needed
  return <div>Combined functionality</div>;
};
```

## 📝 Example Component

See `DataStoreExample.tsx` for a complete working example that demonstrates:
- All store features
- Interactive filtering
- Progress simulation
- Data manipulation
- Real-time updates

## 🎯 Key Benefits

1. **Simple Interface** - Matches your exact specification
2. **Type Safe** - Full TypeScript support
3. **Performance** - Optimized re-rendering
4. **Flexible Filtering** - Multiple filter types
5. **DevTools** - Full debugging support
6. **Extensible** - Easy to add new features

This store provides a clean, simple interface for core data management while maintaining the power and flexibility needed for complex applications.
