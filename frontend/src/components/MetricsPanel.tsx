import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Chip,
  ToggleButton,
  ToggleButtonGroup,
  FormControl,
  Select,
  MenuItem,
  InputLabel,
  Alert,
  Tooltip,
  IconButton,
} from '@mui/material';
import { DataGrid, GridColDef, GridRenderCellParams } from '@mui/x-data-grid';
import {
  TrendingUp,
  Assessment,
  Compare,
  Info,
  Download,
} from '@mui/icons-material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts';
import { useClusteringStore } from '../store';

interface MetricData {
  algorithm: string;
  silhouette_score: number;
  davies_bouldin_score: number;
  calinski_harabasz_score: number;
  inertia?: number;
  n_clusters: number;
  execution_time: number;
}

const MetricsPanel: React.FC = () => {
  const { currentResult, parameters, experiments } = useClusteringStore();
  
  const [viewMode, setViewMode] = useState<'table' | 'chart' | 'radar'>('table');
  const [selectedMetric, setSelectedMetric] = useState<string>('silhouette_score');
  const [comparisonMode, setComparisonMode] = useState<'current' | 'experiments'>('current');

  // Mock metrics data - in real app this would come from currentResult.metrics
  const mockMetrics: MetricData[] = [
    {
      algorithm: 'kmeans',
      silhouette_score: 0.65,
      davies_bouldin_score: 0.89,
      calinski_harabasz_score: 234.5,
      inertia: 152.3,
      n_clusters: parameters.n_clusters,
      execution_time: 2.3,
    },
    {
      algorithm: 'spectral',
      silhouette_score: 0.71,
      davies_bouldin_score: 0.75,
      calinski_harabasz_score: 289.1,
      n_clusters: parameters.n_clusters,
      execution_time: 5.7,
    },
    {
      algorithm: 'dbscan',
      silhouette_score: 0.58,
      davies_bouldin_score: 1.12,
      calinski_harabasz_score: 187.9,
      n_clusters: 4, // DBSCAN finds its own number of clusters
      execution_time: 3.1,
    },
  ];

  const getMetricInfo = (metric: string) => {
    const info = {
      silhouette_score: {
        name: 'Silhouette Score',
        description: 'Measures how similar points are to their own cluster vs other clusters',
        range: '[-1, 1]',
        better: 'higher',
        color: '#1f77b4',
      },
      davies_bouldin_score: {
        name: 'Davies-Bouldin Score',
        description: 'Average similarity ratio of each cluster with its most similar cluster',
        range: '[0, ∞]',
        better: 'lower',
        color: '#ff7f0e',
      },
      calinski_harabasz_score: {
        name: 'Calinski-Harabasz Score',
        description: 'Ratio of sum of between-clusters dispersion and within-cluster dispersion',
        range: '[0, ∞]',
        better: 'higher',
        color: '#2ca02c',
      },
      inertia: {
        name: 'Inertia',
        description: 'Sum of squared distances of samples to their closest cluster center',
        range: '[0, ∞]',
        better: 'lower',
        color: '#d62728',
      },
      execution_time: {
        name: 'Execution Time',
        description: 'Time taken to complete the clustering algorithm',
        range: '[0, ∞]',
        better: 'lower',
        color: '#9467bd',
      },
    };
    return info[metric as keyof typeof info];
  };

  const getScoreColor = (metric: string, value: number) => {
    const info = getMetricInfo(metric);
    if (!info) return 'default';
    
    // Normalize scores to determine color intensity
    if (info.better === 'higher') {
      if (value > 0.7) return 'success';
      if (value > 0.4) return 'warning';
      return 'error';
    } else {
      if (value < 0.5) return 'success';
      if (value < 1.0) return 'warning';
      return 'error';
    }
  };

  const formatMetricValue = (metric: string, value: number) => {
    if (metric === 'execution_time') {
      return `${value.toFixed(2)}s`;
    }
    return value.toFixed(3);
  };

  const getChartData = () => {
    return mockMetrics.map(m => ({
      algorithm: m.algorithm.toUpperCase(),
      [selectedMetric]: m[selectedMetric as keyof MetricData] as number,
    }));
  };

  const getRadarData = () => {
    return mockMetrics.map(metric => ({
      algorithm: metric.algorithm.toUpperCase(),
      silhouette: (metric.silhouette_score + 1) * 50, // Normalize to 0-100
      davies_bouldin: Math.max(0, 100 - metric.davies_bouldin_score * 50), // Invert and normalize
      calinski_harabasz: Math.min(100, metric.calinski_harabasz_score / 5), // Normalize
      speed: Math.max(0, 100 - metric.execution_time * 10), // Invert for speed
    }));
  };

  const exportMetrics = () => {
    console.log('Exporting metrics data...');
  };

  if (!currentResult && experiments.length === 0) {
    return (
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="h6" color="text.secondary">
          Run clustering analysis to see evaluation metrics
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Evaluation Metrics
      </Typography>
      
      {/* Controls */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, alignItems: 'center' }}>
            {/* View Mode Toggle */}
            <ToggleButtonGroup
              value={viewMode}
              exclusive
              onChange={(_, value) => value && setViewMode(value)}
              size="small"
              aria-label="Select metrics view mode"
            >
              <ToggleButton value="table" aria-label="Switch to table view" tabIndex={0}>
                <Assessment sx={{ mr: 1 }} />
                Table
              </ToggleButton>
              <ToggleButton value="chart" aria-label="Switch to chart view" tabIndex={0}>
                <TrendingUp sx={{ mr: 1 }} />
                Chart
              </ToggleButton>
              <ToggleButton value="radar" aria-label="Switch to radar chart view" tabIndex={0}>
                <Compare sx={{ mr: 1 }} />
                Radar
              </ToggleButton>
            </ToggleButtonGroup>

            {/* Comparison Mode */}
            <ToggleButtonGroup
              value={comparisonMode}
              exclusive
              onChange={(_, value) => value && setComparisonMode(value)}
              size="small"
              aria-label="Select comparison mode"
            >
              <ToggleButton value="current" aria-label="Show current run metrics only" tabIndex={0}>Current Run</ToggleButton>
              <ToggleButton value="experiments" aria-label="Show all experiments comparison" tabIndex={0}>All Experiments</ToggleButton>
            </ToggleButtonGroup>

            {/* Metric Selection for Chart View */}
            {viewMode === 'chart' && (
              <FormControl size="small" sx={{ minWidth: 200 }}>
                <InputLabel>Metric</InputLabel>
                <Select
                  value={selectedMetric}
                  onChange={(e) => setSelectedMetric(e.target.value)}
                  aria-label="Select metric to display in chart"
                  tabIndex={0}
                >
                  <MenuItem value="silhouette_score">Silhouette Score</MenuItem>
                  <MenuItem value="davies_bouldin_score">Davies-Bouldin Score</MenuItem>
                  <MenuItem value="calinski_harabasz_score">Calinski-Harabasz Score</MenuItem>
                  <MenuItem value="execution_time">Execution Time</MenuItem>
                </Select>
              </FormControl>
            )}

            {/* Export Button */}
            <Box sx={{ ml: 'auto' }}>
              <Tooltip title="Export Metrics">
                <IconButton 
                  onClick={exportMetrics} 
                  size="small"
                  aria-label="Export metrics data"
                  tabIndex={0}
                >
                  <Download />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>
        </CardContent>
      </Card>

      {/* Metrics Display */}
      <Card>
        <CardContent>
          {viewMode === 'table' && (
            <Box sx={{ height: 600, width: '100%' }}>
              <DataGrid
                rows={mockMetrics.map((metric, index) => ({ id: index, ...metric }))}
                columns={[
                  {
                    field: 'algorithm',
                    headerName: 'Algorithm',
                    width: 150,
                    renderCell: (params: GridRenderCellParams) => (
                      <Chip label={params.value.toUpperCase()} color="primary" />
                    ),
                  },
                  {
                    field: 'silhouette_score',
                    headerName: 'Silhouette Score',
                    width: 180,
                    align: 'center',
                    headerAlign: 'center',
                    renderHeader: () => (
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        Silhouette Score
                        <Tooltip title={getMetricInfo('silhouette_score')?.description}>
                          <IconButton size="small">
                            <Info fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    ),
                    renderCell: (params: GridRenderCellParams) => (
                      <Chip
                        label={formatMetricValue('silhouette_score', params.value)}
                        color={getScoreColor('silhouette_score', params.value)}
                        size="small"
                      />
                    ),
                  },
                  {
                    field: 'davies_bouldin_score',
                    headerName: 'Davies-Bouldin Score',
                    width: 200,
                    align: 'center',
                    headerAlign: 'center',
                    renderHeader: () => (
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        Davies-Bouldin Score
                        <Tooltip title={getMetricInfo('davies_bouldin_score')?.description}>
                          <IconButton size="small">
                            <Info fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    ),
                    renderCell: (params: GridRenderCellParams) => (
                      <Chip
                        label={formatMetricValue('davies_bouldin_score', params.value)}
                        color={getScoreColor('davies_bouldin_score', params.value)}
                        size="small"
                      />
                    ),
                  },
                  {
                    field: 'calinski_harabasz_score',
                    headerName: 'Calinski-Harabasz Score',
                    width: 220,
                    align: 'center',
                    headerAlign: 'center',
                    renderHeader: () => (
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        Calinski-Harabasz Score
                        <Tooltip title={getMetricInfo('calinski_harabasz_score')?.description}>
                          <IconButton size="small">
                            <Info fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    ),
                    renderCell: (params: GridRenderCellParams) => (
                      <Chip
                        label={formatMetricValue('calinski_harabasz_score', params.value)}
                        color={getScoreColor('calinski_harabasz_score', params.value)}
                        size="small"
                      />
                    ),
                  },
                  {
                    field: 'n_clusters',
                    headerName: 'Clusters',
                    width: 120,
                    align: 'center',
                    headerAlign: 'center',
                  },
                  {
                    field: 'execution_time',
                    headerName: 'Time (s)',
                    width: 120,
                    align: 'center',
                    headerAlign: 'center',
                    renderCell: (params: GridRenderCellParams) => 
                      formatMetricValue('execution_time', params.value),
                  },
                ] as GridColDef[]}
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
          )}

          {viewMode === 'chart' && (
            <Box sx={{ height: 400, width: '100%' }}>
              <Typography variant="h6" gutterBottom>
                {getMetricInfo(selectedMetric)?.name} Comparison
              </Typography>
              <ResponsiveContainer>
                <BarChart data={getChartData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="algorithm" />
                  <YAxis />
                  <RechartsTooltip />
                  <Legend />
                  <Bar
                    dataKey={selectedMetric}
                    fill={getMetricInfo(selectedMetric)?.color || '#8884d8'}
                    name={getMetricInfo(selectedMetric)?.name}
                  />
                </BarChart>
              </ResponsiveContainer>
            </Box>
          )}

          {viewMode === 'radar' && (
            <Box sx={{ height: 400, width: '100%' }}>
              <Typography variant="h6" gutterBottom>
                Overall Performance Comparison
              </Typography>
              <ResponsiveContainer>
                <RadarChart data={getRadarData()}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="algorithm" />
                  <PolarRadiusAxis angle={90} domain={[0, 100]} />
                  <Radar
                    name="Silhouette"
                    dataKey="silhouette"
                    stroke="#1f77b4"
                    fill="#1f77b4"
                    fillOpacity={0.1}
                  />
                  <Radar
                    name="Davies-Bouldin (inv)"
                    dataKey="davies_bouldin"
                    stroke="#ff7f0e"
                    fill="#ff7f0e"
                    fillOpacity={0.1}
                  />
                  <Radar
                    name="Calinski-Harabasz"
                    dataKey="calinski_harabasz"
                    stroke="#2ca02c"
                    fill="#2ca02c"
                    fillOpacity={0.1}
                  />
                  <Radar
                    name="Speed"
                    dataKey="speed"
                    stroke="#d62728"
                    fill="#d62728"
                    fillOpacity={0.1}
                  />
                  <Legend />
                </RadarChart>
              </ResponsiveContainer>
            </Box>
          )}
        </CardContent>
      </Card>

      {!currentResult && (
        <Alert severity="info" sx={{ mt: 2 }}>
          This is a preview with mock data. Run clustering analysis to see actual evaluation metrics.
        </Alert>
      )}
    </Box>
  );
};

export default MetricsPanel;
