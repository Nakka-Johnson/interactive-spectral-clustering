/**
 * ExperimentDashboard component for Interactive Spectral Clustering Platform.
 * 
 * Displays a table of clustering runs with their status, metrics, and actions.
 * Provides filtering, sorting, and management capabilities for experiments.
 */

import React, { useEffect, useState, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  IconButton,
  Button,
  Tooltip,
  LinearProgress,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Stack,
  Alert
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  Visibility,
  Download,
  Delete,
  Refresh,
  FilterList
} from '@mui/icons-material';
import { useAppStore, useDatasets, useRuns, useLoading } from '../../store/appStore';
import type { ClusteringRunState } from '../../store/appStore';
import type { AlgorithmId, RunStatus } from '../../types/api';
import { logger } from '../../utils/logger';

/**
 * Props for ExperimentDashboard component.
 */
interface ExperimentDashboardProps {
  onViewRun?: (run: ClusteringRunState) => void;
  onStartRun?: () => void;
  onDeleteRun?: (runId: string) => void;
}

/**
 * Status color mapping for run status chips.
 */
const getStatusColor = (status: RunStatus): "default" | "primary" | "secondary" | "error" | "info" | "success" | "warning" => {
  switch (status) {
    case 'queued': return 'default';
    case 'running': return 'primary';
    case 'done': return 'success';
    case 'error': return 'error';
    default: return 'default';
  }
};

/**
 * Algorithm display names.
 */
const getAlgorithmName = (algorithm: AlgorithmId): string => {
  switch (algorithm) {
    case 'spectral': return 'Spectral';
    case 'kmeans': return 'K-Means';
    case 'dbscan': return 'DBSCAN';
    case 'gmm': return 'GMM';
    case 'agglomerative': return 'Agglomerative';
    default: return algorithm;
  }
};

/**
 * Format duration from start to end time.
 */
const formatDuration = (startTime?: string, endTime?: string): string => {
  if (!startTime) return '-';
  
  const start = new Date(startTime);
  const end = endTime ? new Date(endTime) : new Date();
  const durationMs = end.getTime() - start.getTime();
  
  if (durationMs < 1000) return '< 1s';
  if (durationMs < 60000) return `${Math.round(durationMs / 1000)}s`;
  if (durationMs < 3600000) return `${Math.round(durationMs / 60000)}m`;
  return `${Math.round(durationMs / 3600000)}h`;
};

/**
 * Format metrics for display.
 */
const formatMetrics = (metrics: any): string => {
  if (!metrics) return '-';
  
  const parts: string[] = [];
  if (metrics.silhouette_score !== undefined) {
    parts.push(`Silhouette: ${metrics.silhouette_score.toFixed(3)}`);
  }
  if (metrics.calinski_harabasz_score !== undefined) {
    parts.push(`CH: ${metrics.calinski_harabasz_score.toFixed(1)}`);
  }
  if (metrics.davies_bouldin_score !== undefined) {
    parts.push(`DB: ${metrics.davies_bouldin_score.toFixed(3)}`);
  }
  
  return parts.length > 0 ? parts.join(', ') : '-';
};

/**
 * ExperimentDashboard component.
 */
export const ExperimentDashboard: React.FC<ExperimentDashboardProps> = ({
  onViewRun,
  onStartRun,
  onDeleteRun
}) => {
  // Store state
  const { fetchRuns, fetchDatasets, setActiveDataset } = useAppStore();
  const datasets = useDatasets();
  const runs = useRuns();
  const isLoading = useLoading();
  
  // Local state
  const [selectedDataset, setSelectedDataset] = useState<string>('all');
  const [statusFilter, setStatusFilter] = useState<string>('all');

  // Memoized functions to prevent unnecessary re-renders
  const fetchDatasetsCallback = useCallback(() => {
    fetchDatasets();
  }, [fetchDatasets]);

  const fetchRunsCallback = useCallback(() => {
    fetchRuns();
  }, [fetchRuns]);

  // Load data on mount
  useEffect(() => {
    fetchDatasetsCallback();
    fetchRunsCallback();
  }, [fetchDatasetsCallback, fetchRunsCallback]);

  // Filter runs based on selected filters
  const filteredRuns = runs.filter(run => {
    if (selectedDataset !== 'all' && run.dataset_job_id !== selectedDataset) {
      return false;
    }
    if (statusFilter !== 'all' && run.status !== statusFilter) {
      return false;
    }
    return true;
  });

  // Handle dataset filter change
  const handleDatasetChange = (datasetId: string) => {
    setSelectedDataset(datasetId);
    if (datasetId !== 'all') {
      const dataset = datasets.find(d => d.job_id === datasetId);
      if (dataset) {
        setActiveDataset(dataset);
        fetchRuns(datasetId);
      }
    } else {
      setActiveDataset(null);
      fetchRuns();
    }
  };

  // Handle refresh
  const handleRefresh = () => {
    if (selectedDataset !== 'all') {
      fetchRuns(selectedDataset);
    } else {
      fetchRuns();
    }
  };

  // Handle run actions
  const handleViewRun = (run: ClusteringRunState) => {
    onViewRun?.(run);
  };

  const handleDeleteRun = (runId: string) => {
    if (globalThis.confirm('Are you sure you want to delete this run?')) {
      onDeleteRun?.(runId);
    }
  };

  const handleStopRun = (runId: string) => {
    // TODO: Implement stop run functionality
    logger.info('Stop run:', runId);
  };

  return (
    <Box sx={{ maxWidth: 1200, mx: 'auto', p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1">
          Experiment Dashboard
        </Typography>
        
        <Button
          variant="contained"
          onClick={onStartRun}
          startIcon={<PlayArrow />}
        >
          New Experiment
        </Button>
      </Box>

      <Typography variant="body1" color="text.secondary" paragraph>
        Monitor and manage your clustering experiments. View results, track progress, and compare different algorithm configurations.
      </Typography>

      {/* Filters */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Stack direction="row" spacing={2} alignItems="center">
            <FilterList color="primary" />
            
            <FormControl size="small" sx={{ minWidth: 200 }}>
              <InputLabel>Dataset</InputLabel>
              <Select
                value={selectedDataset}
                label="Dataset"
                onChange={(e) => handleDatasetChange(e.target.value)}
              >
                <MenuItem value="all">All Datasets</MenuItem>
                {datasets.map((dataset) => (
                  <MenuItem key={dataset.job_id} value={dataset.job_id}>
                    {dataset.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControl size="small" sx={{ minWidth: 150 }}>
              <InputLabel>Status</InputLabel>
              <Select
                value={statusFilter}
                label="Status"
                onChange={(e) => setStatusFilter(e.target.value)}
              >
                <MenuItem value="all">All Status</MenuItem>
                <MenuItem value="queued">Queued</MenuItem>
                <MenuItem value="running">Running</MenuItem>
                <MenuItem value="done">Completed</MenuItem>
                <MenuItem value="error">Error</MenuItem>
              </Select>
            </FormControl>

            <Button
              startIcon={<Refresh />}
              onClick={handleRefresh}
              disabled={isLoading}
            >
              Refresh
            </Button>
          </Stack>
        </CardContent>
      </Card>

      {/* Loading indicator */}
      {isLoading && <LinearProgress sx={{ mb: 2 }} />}

      {/* Runs table */}
      <Card>
        <CardContent sx={{ p: 0 }}>
          {filteredRuns.length === 0 ? (
            <Box sx={{ p: 4, textAlign: 'center' }}>
              <Typography variant="h6" color="text.secondary" gutterBottom>
                No experiments found
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                {selectedDataset !== 'all' || statusFilter !== 'all' 
                  ? 'Try adjusting your filters or create a new experiment.'
                  : 'Get started by uploading a dataset and running your first clustering experiment.'
                }
              </Typography>
              {onStartRun && (
                <Button variant="outlined" onClick={onStartRun}>
                  Start First Experiment
                </Button>
              )}
            </Box>
          ) : (
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Run ID</TableCell>
                    <TableCell>Dataset</TableCell>
                    <TableCell>Algorithm</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Started</TableCell>
                    <TableCell>Duration</TableCell>
                    <TableCell>Metrics</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {filteredRuns.map((run) => (
                    <TableRow key={run.run_id} hover>
                      <TableCell>
                        <Typography variant="body2" fontFamily="monospace">
                          {run.run_id.slice(0, 8)}...
                        </Typography>
                      </TableCell>
                      
                      <TableCell>
                        <Typography variant="body2">
                          {run.dataset_name || 'Unknown'}
                        </Typography>
                      </TableCell>
                      
                      <TableCell>
                        <Chip
                          label={getAlgorithmName(run.algorithm)}
                          size="small"
                          variant="outlined"
                        />
                      </TableCell>
                      
                      <TableCell>
                        <Chip
                          label={run.status}
                          color={getStatusColor(run.status)}
                          size="small"
                        />
                        {run.status === 'running' && run.is_loading && (
                          <LinearProgress sx={{ mt: 1, width: 100 }} />
                        )}
                      </TableCell>
                      
                      <TableCell>
                        <Typography variant="body2">
                          {run.started_at 
                            ? new Date(run.started_at).toLocaleString()
                            : '-'
                          }
                        </Typography>
                      </TableCell>
                      
                      <TableCell>
                        <Typography variant="body2">
                          {formatDuration(run.started_at, run.ended_at)}
                        </Typography>
                      </TableCell>
                      
                      <TableCell>
                        <Typography variant="body2" sx={{ maxWidth: 200 }}>
                          {formatMetrics(run.metrics)}
                        </Typography>
                      </TableCell>
                      
                      <TableCell>
                        <Stack direction="row" spacing={1}>
                          {run.status === 'done' && (
                            <Tooltip title="View Results">
                              <IconButton
                                size="small"
                                onClick={() => handleViewRun(run)}
                              >
                                <Visibility />
                              </IconButton>
                            </Tooltip>
                          )}
                          
                          {run.status === 'running' && (
                            <Tooltip title="Stop Run">
                              <IconButton
                                size="small"
                                onClick={() => handleStopRun(run.run_id)}
                              >
                                <Stop />
                              </IconButton>
                            </Tooltip>
                          )}
                          
                          {run.status === 'done' && (
                            <Tooltip title="Download Results">
                              <IconButton size="small">
                                <Download />
                              </IconButton>
                            </Tooltip>
                          )}
                          
                          <Tooltip title="Delete Run">
                            <IconButton
                              size="small"
                              onClick={() => handleDeleteRun(run.run_id)}
                              disabled={run.status === 'running'}
                            >
                              <Delete />
                            </IconButton>
                          </Tooltip>
                        </Stack>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </CardContent>
      </Card>

      {/* Error display */}
      {filteredRuns.some(run => run.status === 'error') && (
        <Alert severity="warning" sx={{ mt: 2 }}>
          Some experiments have failed. Check the individual run details for error messages.
        </Alert>
      )}
    </Box>
  );
};

export default ExperimentDashboard;
