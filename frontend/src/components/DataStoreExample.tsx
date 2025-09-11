import React, { useEffect } from 'react';
import { DataGrid, GridColDef, GridRenderCellParams } from '@mui/x-data-grid';
import { Box, Button } from '@mui/material';
import { 
  useStore, 
  useJobId, 
  useDataRows, 
  useColumns, 
  useFilteredData,
  useIsLoading,
  useHasData,
  useDataActions 
} from '../store/dataStore';

/**
 * Example component demonstrating usage of the new Zustand store slice
 */
const DataStoreExample: React.FC = () => {
  // Method 1: Use the main store hook (gets all state and actions)
  const { 
    dataRows,
    filters, 
    results, 
    progress
  } = useStore();

  // Method 2: Use individual hooks for better performance (only re-renders when specific values change)
  const optimizedJobId = useJobId();
  const optimizedDataRows = useDataRows();
  const optimizedColumns = useColumns();
  
  // Method 3: Use computed value hooks
  const filteredData = useFilteredData();
  const isLoading = useIsLoading();
  const hasData = useHasData();
  
  // Method 4: Use action hooks (for components that only need actions, not state)
  const actions = useDataActions();

  // Example initialization
  useEffect(() => {
    // Initialize with sample data
    actions.setJobId('job_' + Date.now());
    actions.setColumns(['id', 'name', 'age', 'department']);
    actions.setDataRows([
      { id: 1, name: 'John Doe', age: 30, department: 'Engineering' },
      { id: 2, name: 'Jane Smith', age: 25, department: 'Design' },
      { id: 3, name: 'Bob Johnson', age: 35, department: 'Engineering' },
      { id: 4, name: 'Alice Brown', age: 28, department: 'Marketing' },
    ]);
  }, []); // Empty dependency array - only run once on mount

  // Example handlers
  const handleFilterByDepartment = (dept: string) => {
    actions.updateFilter('department', dept);
  };

  const handleFilterByAgeRange = (min: number, max: number) => {
    actions.updateFilter('age', { min, max });
  };

  const handleClearFilters = () => {
    actions.setFilters({});
  };

  const handleSimulateProgress = () => {
    let progress = 0;
    const interval = setInterval(() => {
      progress += 10;
      actions.setProgress(progress);
      if (progress >= 100) {
        clearInterval(interval);
        actions.setResults({
          run_id: `example_run_${Date.now()}`,
          algorithm: 'kmeans',
          parameters: { n_clusters: 3 },
          status: 'completed',
          labels: [0, 1, 2, 0, 1, 2], // Mock cluster assignments
          metrics: {
            silhouette_score: 0.95,
            n_clusters_found: 3
          },
          ended_at: new Date().toISOString()
        });
      }
    }, 500);
  };

  const handleAddRandomPerson = () => {
    const names = ['Charlie', 'Diana', 'Eve', 'Frank', 'Grace'];
    const departments = ['Engineering', 'Design', 'Marketing', 'Sales'];
    const randomName = names[Math.floor(Math.random() * names.length)];
    const randomDept = departments[Math.floor(Math.random() * departments.length)];
    const randomAge = Math.floor(Math.random() * 20) + 25;
    
    actions.addDataRow({
      id: Date.now(),
      name: randomName,
      age: randomAge,
      department: randomDept
    });
  };

  return (
    <div style={{ padding: '20px', maxWidth: '800px', margin: '0 auto' }}>
      <h1>Zustand Data Store Example</h1>
      
      {/* Status Section */}
      <div style={{ marginBottom: '20px', padding: '10px', backgroundColor: '#f5f5f5', borderRadius: '5px' }}>
        <h3>Store Status</h3>
        <p><strong>Job ID:</strong> {optimizedJobId}</p>
        <p><strong>Has Data:</strong> {hasData ? 'Yes' : 'No'}</p>
        <p><strong>Total Rows:</strong> {optimizedDataRows.length}</p>
        <p><strong>Filtered Rows:</strong> {filteredData.length}</p>
        <p><strong>Is Loading:</strong> {isLoading ? 'Yes' : 'No'}</p>
        <p><strong>Progress:</strong> {progress}%</p>
        {results && <p><strong>Results:</strong> {JSON.stringify(results)}</p>}
      </div>

      {/* Controls Section */}
      <div style={{ marginBottom: '20px' }}>
        <h3>Controls</h3>
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', marginBottom: '10px' }}>
          <button onClick={() => handleFilterByDepartment('Engineering')}>
            Filter by Engineering
          </button>
          <button onClick={() => handleFilterByDepartment('Design')}>
            Filter by Design
          </button>
          <button onClick={() => handleFilterByAgeRange(25, 30)}>
            Filter Age 25-30
          </button>
          <button onClick={handleClearFilters}>
            Clear Filters
          </button>
        </div>
        
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
          <button onClick={handleAddRandomPerson}>
            Add Random Person
          </button>
          <button onClick={handleSimulateProgress}>
            Simulate Progress
          </button>
          <button onClick={() => actions.reset()}>
            Reset Store
          </button>
        </div>
      </div>

      {/* Active Filters Section */}
      {Object.keys(filters).length > 0 && (
        <div style={{ marginBottom: '20px' }}>
          <h3>Active Filters</h3>
          <pre style={{ backgroundColor: '#f0f0f0', padding: '10px', borderRadius: '5px' }}>
            {JSON.stringify(filters, null, 2)}
          </pre>
        </div>
      )}

      {/* Data Table */}
      <div>
        <h3>Data ({filteredData.length} rows)</h3>
        <Box sx={{ height: 400, width: '100%' }}>
          <DataGrid
            rows={filteredData.map((row, index) => ({ 
              id: row.id || index, 
              ...row 
            }))}
            columns={[
              ...optimizedColumns.map((column): GridColDef => ({
                field: column,
                headerName: column,
                width: 150,
                flex: 1,
              })),
              {
                field: 'actions',
                headerName: 'Actions',
                width: 120,
                sortable: false,
                disableColumnMenu: true,
                renderCell: (params: GridRenderCellParams) => (
                  <Button
                    size="small"
                    variant="outlined"
                    color="error"
                    onClick={() => {
                      const originalIndex = dataRows.findIndex(r => r.id === params.row.id);
                      if (originalIndex !== -1) {
                        actions.removeDataRow(originalIndex);
                      }
                    }}
                  >
                    Remove
                  </Button>
                ),
              },
            ]}
            initialState={{
              pagination: {
                paginationModel: { pageSize: 50, page: 0 },
              },
            }}
            pageSizeOptions={[50, 100, 500]}
            autoHeight
            disableRowSelectionOnClick
            density="compact"
            sx={{
              '& .MuiDataGrid-cell': {
                borderRight: '1px solid #e0e0e0',
              },
              '& .MuiDataGrid-columnHeaders': {
                backgroundColor: '#f5f5f5',
                borderBottom: '2px solid #e0e0e0',
              },
            }}
          />
        </Box>
      </div>

      {/* Loading Indicator */}
      {isLoading && (
        <div style={{ 
          position: 'fixed', 
          top: '50%', 
          left: '50%', 
          transform: 'translate(-50%, -50%)',
          backgroundColor: 'rgba(0,0,0,0.8)',
          color: 'white',
          padding: '20px',
          borderRadius: '5px',
          zIndex: 1000
        }}>
          <div>Loading... {progress}%</div>
          <div style={{ 
            width: '200px', 
            height: '10px', 
            backgroundColor: '#ccc', 
            borderRadius: '5px',
            marginTop: '10px'
          }}>
            <div style={{ 
              width: `${progress}%`, 
              height: '100%', 
              backgroundColor: '#007bff',
              borderRadius: '5px',
              transition: 'width 0.3s ease'
            }} />
          </div>
        </div>
      )}
    </div>
  );
};

export default DataStoreExample;
