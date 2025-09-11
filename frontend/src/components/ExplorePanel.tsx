import React, { useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Chip,
  LinearProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent,
} from '@mui/material';
import {
  TableChart,
  ViewColumn,
  Functions,
  Storage,
} from '@mui/icons-material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
} from 'recharts';
import { useClusteringStore } from '../store';

interface ColumnStat {
  name: string;
  mean: number;
  std: number;
  missing: number;
  missingPercent: number;
  min: number;
  max: number;
  count: number;
}

const ExplorePanel: React.FC = () => {
  const { dataset, filters } = useClusteringStore();
  const [selectedColumns, setSelectedColumns] = React.useState<string[]>([]);

  // Generate mock statistics for demonstration
  const columnStats: ColumnStat[] = useMemo(() => {
    if (!dataset) return [];
    
    return dataset.numericColumns.map((col, index) => ({
      name: col,
      mean: Math.random() * 100,
      std: Math.random() * 20,
      missing: Math.floor(Math.random() * dataset.shape[0] * 0.1),
      missingPercent: Math.random() * 10,
      min: Math.random() * 10,
      max: 90 + Math.random() * 10,
      count: dataset.shape[0] - Math.floor(Math.random() * dataset.shape[0] * 0.1),
    }));
  }, [dataset]);

  // Generate mock scatter plot data
  const scatterData = useMemo(() => {
    if (selectedColumns.length < 2) return [];
    
    return Array.from({ length: 100 }, (_, i) => ({
      x: Math.random() * 100,
      y: Math.random() * 100,
      id: i,
    }));
  }, [selectedColumns]);

  const handleColumnSelection = (event: SelectChangeEvent<string[]>) => {
    const value = event.target.value;
    setSelectedColumns(typeof value === 'string' ? value.split(',') : value);
  };

  if (!dataset) {
    return (
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="h6" color="text.secondary">
          Upload a dataset to explore its characteristics
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Data Exploration
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Explore statistical properties and relationships in your dataset
      </Typography>

      {/* Dataset Overview */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Dataset Overview
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
            <Card variant="outlined" sx={{ flex: '1 1 200px', minWidth: 200 }}>
              <CardContent sx={{ textAlign: 'center' }}>
                <TableChart color="primary" sx={{ fontSize: 40, mb: 1 }} />
                <Typography variant="h4" color="primary">
                  {dataset.shape[0].toLocaleString()}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Rows
                </Typography>
              </CardContent>
            </Card>

            <Card variant="outlined" sx={{ flex: '1 1 200px', minWidth: 200 }}>
              <CardContent sx={{ textAlign: 'center' }}>
                <ViewColumn color="secondary" sx={{ fontSize: 40, mb: 1 }} />
                <Typography variant="h4" color="secondary">
                  {dataset.numericColumns.length}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Numeric Columns
                </Typography>
              </CardContent>
            </Card>

            <Card variant="outlined" sx={{ flex: '1 1 200px', minWidth: 200 }}>
              <CardContent sx={{ textAlign: 'center' }}>
                <Functions color="success" sx={{ fontSize: 40, mb: 1 }} />
                <Typography variant="h4" color="success.main">
                  {filters.columnFilters?.length || 0}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Selected for Analysis
                </Typography>
              </CardContent>
            </Card>

            <Card variant="outlined" sx={{ flex: '1 1 200px', minWidth: 200 }}>
              <CardContent sx={{ textAlign: 'center' }}>
                <Storage color="info" sx={{ fontSize: 40, mb: 1 }} />
                <Typography variant="h4" color="info.main">
                  {dataset.columns.length - dataset.numericColumns.length}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Categorical Columns
                </Typography>
              </CardContent>
            </Card>
          </Box>
        </CardContent>
      </Card>

      {/* Column Statistics */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Column Statistics
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Statistical summary of numeric columns
          </Typography>

          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
            {columnStats.slice(0, 6).map((stat) => (
              <Card variant="outlined" key={stat.name} sx={{ flex: '1 1 300px', minWidth: 300 }}>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                      {stat.name}
                    </Typography>
                    <Chip
                      size="small"
                      label={filters.columnFilters?.includes(stat.name) ? 'Selected' : 'Available'}
                      color={filters.columnFilters?.includes(stat.name) ? 'primary' : 'default'}
                      sx={{ ml: 1 }}
                    />
                  </Box>
                    
                    <Box sx={{ mb: 2 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                        <Typography variant="body2">Mean:</Typography>
                        <Typography variant="body2" fontWeight="bold">
                          {stat.mean.toFixed(2)}
                        </Typography>
                      </Box>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                        <Typography variant="body2">Std Dev:</Typography>
                        <Typography variant="body2" fontWeight="bold">
                          {stat.std.toFixed(2)}
                        </Typography>
                      </Box>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                        <Typography variant="body2">Range:</Typography>
                        <Typography variant="body2" fontWeight="bold">
                          {stat.min.toFixed(1)} - {stat.max.toFixed(1)}
                        </Typography>
                      </Box>
                    </Box>

                    <Box>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography variant="body2">Missing Values:</Typography>
                        <Typography variant="body2" color={stat.missingPercent > 5 ? 'error' : 'text.secondary'}>
                          {stat.missing} ({stat.missingPercent.toFixed(1)}%)
                        </Typography>
                      </Box>
                      <LinearProgress 
                        variant="determinate" 
                        value={100 - stat.missingPercent}
                        color={stat.missingPercent > 5 ? 'error' : 'primary'}
                        sx={{ height: 4, borderRadius: 2 }}
                      />
                    </Box>
                  </CardContent>
                </Card>
            ))}
          </Box>

          {columnStats.length > 6 && (
            <Box sx={{ mt: 2, textAlign: 'center' }}>
              <Typography variant="body2" color="text.secondary">
                Showing 6 of {columnStats.length} columns
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Pair Plot Preview */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Feature Relationships
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Select columns to visualize their relationships
          </Typography>

          <FormControl fullWidth sx={{ mb: 3 }}>
            <InputLabel>Select Columns for Scatter Plot</InputLabel>
            <Select
              multiple
              value={selectedColumns}
              onChange={handleColumnSelection}
              aria-label="Select columns to visualize in scatter plot"
              tabIndex={0}
              renderValue={(selected) => (
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                  {selected.map((value) => (
                    <Chip key={value} label={value} size="small" />
                  ))}
                </Box>
              )}
            >
              {dataset.numericColumns.map((column) => (
                <MenuItem key={column} value={column}>
                  {column}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {selectedColumns.length >= 2 && (
            <Box sx={{ height: 400 }}>
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart data={scatterData} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    type="number" 
                    dataKey="x" 
                    name={selectedColumns[0]}
                    label={{ value: selectedColumns[0], position: 'insideBottom', offset: -10 }}
                  />
                  <YAxis 
                    type="number" 
                    dataKey="y" 
                    name={selectedColumns[1]}
                    label={{ value: selectedColumns[1], angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip 
                    cursor={{ strokeDasharray: '3 3' }}
                    formatter={(value: any, name: string) => [
                      typeof value === 'number' ? value.toFixed(2) : value,
                      name
                    ]}
                  />
                  <Scatter name="Data Points" dataKey="y" fill="#8884d8" />
                </ScatterChart>
              </ResponsiveContainer>
            </Box>
          )}

          {selectedColumns.length < 2 && (
            <Box sx={{ 
              height: 200, 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center',
              bgcolor: 'grey.50',
              borderRadius: 1,
              border: '1px dashed',
              borderColor: 'grey.300'
            }}>
              <Typography variant="body1" color="text.secondary">
                Select at least 2 columns to view scatter plot
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Distribution Preview */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Data Distribution
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Distribution of values across selected columns
          </Typography>

          {filters.columnFilters && filters.columnFilters.length > 0 && (
            <Box sx={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart 
                  data={columnStats.filter(stat => filters.columnFilters?.includes(stat.name))}
                  margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="name" 
                    angle={-45}
                    textAnchor="end"
                    height={80}
                    interval={0}
                  />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="mean" fill="#8884d8" name="Mean Value" />
                </BarChart>
              </ResponsiveContainer>
            </Box>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};

export default ExplorePanel;
