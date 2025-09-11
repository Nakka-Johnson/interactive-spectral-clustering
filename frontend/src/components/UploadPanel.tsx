import React, { useState, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Alert,
  LinearProgress,
  Chip,
  FormControlLabel,
  Switch,
} from '@mui/material';
import { DataGrid, GridColDef } from '@mui/x-data-grid';
import {
  CloudUpload,
  Description,
  FilterList,
  Visibility,
  VisibilityOff,
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import { useClusteringStore } from '../store';
import { uploadData, formatApiError } from '../api';

interface UploadPanelProps {}

const UploadPanel: React.FC<UploadPanelProps> = () => {
  const {
    dataset,
    filters,
    setDataset,
    setFilters,
    setError,
    clearError,
    setCurrentTab,
  } = useClusteringStore();

  const [uploading, setUploading] = useState(false);
  const [previewData, setPreviewData] = useState<any[]>([]);
  const [showAllColumns, setShowAllColumns] = useState(false);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    // Validate file type
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setError('Please upload a CSV file');
      return;
    }

    setUploading(true);
    clearError();

    try {
      const response = await uploadData(file);
      
      // Create dataset object
      const newDataset = {
        jobId: response.job_id,
        filename: file.name,
        columns: response.columns,
        numericColumns: response.numeric_columns,
        shape: response.shape,
        uploadTime: new Date(),
      };

      setDataset(newDataset);
      
      // Initialize filters
      setFilters({
        columnFilters: response.numeric_columns,
        useAllColumns: true,
        rowConditions: {},
      });

      // Generate preview data (mock for now)
      generatePreviewData(response.columns);
      
      // Auto-navigate to explore tab
      setCurrentTab('explore');
      
    } catch (error: any) {
      setError(formatApiError(error));
    } finally {
      setUploading(false);
    }
  }, [setDataset, setFilters, setError, clearError, setCurrentTab]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
    },
    multiple: false,
    disabled: uploading,
  });

  const generatePreviewData = (columns: string[]) => {
    // Generate mock preview data
    const mockData = Array.from({ length: 5 }, (_, index) => {
      const row: any = { _id: index };
      columns.forEach((col) => {
        if (col.toLowerCase().includes('id')) {
          row[col] = `ID_${index + 1}`;
        } else if (Math.random() > 0.5) {
          row[col] = (Math.random() * 100).toFixed(2);
        } else {
          row[col] = ['A', 'B', 'C'][Math.floor(Math.random() * 3)];
        }
      });
      return row;
    });
    setPreviewData(mockData);
  };

  const handleColumnFilterChange = (selectedColumns: string[]) => {
    setFilters({
      ...filters,
      columnFilters: selectedColumns,
      useAllColumns: selectedColumns.length === dataset?.numericColumns.length,
    });
  };

  const toggleColumnVisibility = (column: string) => {
    if (!dataset || !filters.columnFilters) return;
    
    const isSelected = filters.columnFilters.includes(column);
    const newSelection = isSelected
      ? filters.columnFilters.filter(col => col !== column)
      : [...filters.columnFilters, column];
    
    handleColumnFilterChange(newSelection);
  };

  const displayColumns = showAllColumns 
    ? dataset?.columns || []
    : dataset?.columns.slice(0, 8) || [];

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Upload Data
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Upload a CSV file to begin clustering analysis
      </Typography>

      {/* File Upload Area */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box
            {...getRootProps()}
            sx={{
              border: '2px dashed',
              borderColor: isDragActive ? 'primary.main' : 'grey.300',
              borderRadius: 2,
              p: 6,
              textAlign: 'center',
              cursor: 'pointer',
              bgcolor: isDragActive ? 'action.hover' : 'background.paper',
              transition: 'all 0.2s ease',
              '&:hover': {
                bgcolor: 'action.hover',
                borderColor: 'primary.main',
              },
            }}
          >
            <input {...getInputProps()} />
            <CloudUpload sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              {isDragActive ? 'Drop the CSV file here' : 'Drag & drop a CSV file here'}
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              or click to select a file
            </Typography>
            <Button 
              variant="outlined" 
              disabled={uploading}
              aria-label="Choose CSV file to upload"
              tabIndex={0}
            >
              Choose File
            </Button>
          </Box>
          
          {uploading && (
            <Box sx={{ mt: 2 }}>
              <LinearProgress />
              <Typography variant="body2" sx={{ mt: 1, textAlign: 'center' }}>
                Uploading and processing file...
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Dataset Information */}
      {dataset && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              <Description sx={{ mr: 1, verticalAlign: 'middle' }} />
              Dataset Information
            </Typography>
            
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
              <Chip label={`${dataset.shape[0]} rows`} color="primary" />
              <Chip label={`${dataset.shape[1]} columns`} color="secondary" />
              <Chip label={`${dataset.numericColumns.length} numeric`} color="success" />
              <Chip 
                label={`${dataset.columns.length - dataset.numericColumns.length} categorical`} 
                color="info" 
              />
            </Box>

            <Typography variant="body2" color="text.secondary">
              File: {dataset.filename} • Uploaded: {dataset.uploadTime.toLocaleString()}
            </Typography>
          </CardContent>
        </Card>
      )}

      {/* Column Filter */}
      {dataset && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <FilterList sx={{ mr: 1 }} />
              <Typography variant="h6">
                Column Selection
              </Typography>
              <Box sx={{ ml: 'auto' }}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={showAllColumns}
                      onChange={(e) => setShowAllColumns(e.target.checked)}
                      aria-label="Toggle show all columns visibility"
                      tabIndex={0}
                    />
                  }
                  label="Show all columns"
                />
              </Box>
            </Box>

            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Select columns to include in clustering analysis (only numeric columns can be used)
            </Typography>

            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {displayColumns.map((column) => {
                const isNumeric = dataset.numericColumns.includes(column);
                const isSelected = filters.columnFilters?.includes(column);
                
                return (
                  <Chip
                    key={column}
                    label={column}
                    color={isSelected ? 'primary' : 'default'}
                    variant={isSelected ? 'filled' : 'outlined'}
                    disabled={!isNumeric}
                    onClick={() => isNumeric && toggleColumnVisibility(column)}
                    aria-label={`${isSelected ? 'Remove' : 'Add'} ${column} column ${isNumeric ? 'for clustering' : '(non-numeric, disabled)'}`}
                    tabIndex={isNumeric ? 0 : -1}
                    icon={
                      isNumeric ? (
                        isSelected ? <Visibility /> : <VisibilityOff />
                      ) : undefined
                    }
                    sx={{
                      cursor: isNumeric ? 'pointer' : 'default',
                      opacity: isNumeric ? 1 : 0.5,
                    }}
                  />
                );
              })}
              
              {!showAllColumns && dataset.columns.length > 8 && (
                <Chip
                  label={`+${dataset.columns.length - 8} more`}
                  variant="outlined"
                  onClick={() => setShowAllColumns(true)}
                  aria-label={`Show ${dataset.columns.length - 8} more columns`}
                  tabIndex={0}
                  sx={{ cursor: 'pointer' }}
                />
              )}
            </Box>

            {filters.columnFilters && (
              <Alert severity="info" sx={{ mt: 2 }}>
                {filters.columnFilters.length} columns selected for clustering
              </Alert>
            )}
          </CardContent>
        </Card>
      )}

      {/* Data Preview */}
      {dataset && previewData.length > 0 && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Data Preview
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              First 5 rows of your dataset
            </Typography>
            
            <Box sx={{ height: 400, width: '100%' }}>
              <DataGrid
                rows={previewData.map((row, index) => ({ id: index, ...row }))}
                columns={displayColumns.map((field): GridColDef => ({
                  field,
                  headerName: field,
                  width: 150,
                  flex: 1,
                  renderHeader: () => (
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      {field}
                      {dataset.numericColumns.includes(field) && (
                        <Chip size="small" label="Numeric" sx={{ ml: 1 }} />
                      )}
                    </Box>
                  ),
                }))}
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
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default UploadPanel;
