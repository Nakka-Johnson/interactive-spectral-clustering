/**
 * VisualizePage component for Interactive Spectral Clustering Platform.
 * 
 * Combines the ControlsBar and Cluster2D components to provide a complete
 * visualization interface for clustering configuration and results display.
 */

import React, { useState, useCallback } from 'react';
import {
  Box,
  Typography,
  Alert,
  Card,
  CardContent,
  CircularProgress,
  Stack
} from '@mui/material';
import ControlsBar from '../features/visualize/ControlsBar';
import { Cluster2D } from '../features/visualize/Cluster2D';
import { useAppStore, useActiveDataset, useActiveRun } from '../store/appStore';
import type { AlgorithmId, ParameterMap, RunStatus } from '../types/api';

/**
 * Mock data for demonstration purposes.
 */
const generateMockData = (numPoints: number = 150) => {
  const points: number[][] = [];
  const labels: number[] = [];
  
  // Generate 3 clusters
  for (let cluster = 0; cluster < 3; cluster++) {
    const centerX = (cluster - 1) * 3;
    const centerY = (cluster % 2 === 0 ? 1 : -1) * 2;
    
    for (let i = 0; i < numPoints / 3; i++) {
      const angle = Math.random() * 2 * Math.PI;
      const radius = Math.random() * 1.5;
      const x = centerX + radius * Math.cos(angle) + (Math.random() - 0.5) * 0.5;
      const y = centerY + radius * Math.sin(angle) + (Math.random() - 0.5) * 0.5;
      
      points.push([x, y]);
      labels.push(cluster);
    }
  }
  
  return { points, labels };
};

/**
 * VisualizePage component.
 */
export const VisualizePage: React.FC = () => {
  // Store state - use optimized selectors to avoid infinite renders
  const { addRun, updateRun } = useAppStore();
  const activeDataset = useActiveDataset();
  const activeRun = useActiveRun();
  
  // Local state
  const [isRunning, setIsRunning] = useState(false);
  const [currentAlgorithm, setCurrentAlgorithm] = useState<AlgorithmId>('spectral' as AlgorithmId);
  const [visualizationData, setVisualizationData] = useState<{
    points: number[][];
    labels: number[];
  } | null>(null);

  // Handle clustering run
  const handleRunClustering = useCallback(async (config: {
    datasetId: string;
    algorithm: AlgorithmId;
    parameters: ParameterMap;
  }) => {
    setIsRunning(true);
    
    // Generate a unique run ID
    const runId = `run_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    try {
      // Create new run in store
      const startTime = new Date().toISOString();
      addRun({
        run_id: runId,
        dataset_job_id: config.datasetId,
        algorithm: config.algorithm,
        parameters: config.parameters,
        status: 'running' as RunStatus,
        started_at: startTime,
        dataset_name: activeDataset?.name,
        is_loading: true,
        // Required computed properties
        id: runId,
        dataset: config.datasetId || activeDataset?.name || 'unknown',
        createdAt: startTime
      });

      // Simulate clustering process
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Generate mock results
      const mockData = generateMockData();
      
      // Update run with results
      updateRun(runId, {
        status: 'completed' as RunStatus,
        ended_at: new Date().toISOString(),
        labels: mockData.labels,
        metrics: {
          silhouette_score: 0.75 + Math.random() * 0.2,
          davies_bouldin_score: 0.5 + Math.random() * 0.3,
          calinski_harabasz_score: 200 + Math.random() * 400,
          inertia: 50 + Math.random() * 100
        },
        execution_time_seconds: 2,
        gpu_used: config.parameters.use_gpu || false,
        is_loading: false
      });
      
      // Update visualization
      setVisualizationData(mockData);
      
    } catch (error) {
      console.error('Clustering failed:', error);
      updateRun(runId, {
        status: 'error' as any,
        ended_at: new Date().toISOString(),
        error_message: 'Clustering simulation failed',
        is_loading: false
      });
    } finally {
      setIsRunning(false);
    }
  }, [activeDataset, addRun, updateRun]);

  // Handle parameter changes
  const handleParametersChange = useCallback((algorithm: AlgorithmId, parameters: ParameterMap) => {
    setCurrentAlgorithm(algorithm);
    // Parameters are handled by the ControlsBar component
  }, []);

  // Generate visualization data when activeRun changes
  const updateVisualizationData = useCallback(() => {
    if (activeRun && activeRun.status === 'done' && activeRun.labels) {
      // For now, generate mock 2D coordinates
      // In a real implementation, this would come from the embedding/dimensionality reduction
      const mockData = generateMockData(activeRun.labels.length);
      const nextData = {
        points: mockData.points,
        labels: activeRun.labels
      };
      setVisualizationData(prev => {
        // Use shallow comparison to prevent unnecessary updates
        if (!prev) return nextData;
        if (prev.labels === activeRun.labels && prev.points.length === nextData.points.length) {
          return prev;
        }
        return nextData;
      });
    } else if (!activeRun || activeRun.status !== 'done') {
      // Clear visualization if run is not complete
      setVisualizationData(prev => prev === null ? prev : null);
    }
  }, [activeRun]);

  React.useEffect(() => {
    updateVisualizationData();
  }, [updateVisualizationData]);

  return (
    <Box sx={{ p: 3, height: '100%' }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Clustering Visualization
      </Typography>
      
      <Typography variant="body1" color="text.secondary" paragraph>
        Configure clustering parameters and visualize results in real-time.
        Select a dataset, choose an algorithm, adjust parameters, and run clustering to see the 2D visualization.
      </Typography>

      <Box sx={{ 
        display: 'flex', 
        gap: 3, 
        height: 'calc(100% - 140px)',
        flexDirection: { xs: 'column', md: 'row' }
      }}>
        {/* Controls Panel */}
        <Box sx={{ 
          width: { xs: '100%', md: '350px' },
          flexShrink: 0
        }}>
          <ControlsBar
            onRunClustering={handleRunClustering}
            onParametersChange={handleParametersChange}
            disabled={isRunning}
          />
        </Box>

        {/* Visualization Panel */}
        <Box sx={{ flex: 1, minHeight: '500px' }}>
          <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <CardContent sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
              {!activeDataset ? (
                <Alert severity="info" sx={{ mb: 2 }}>
                  Please select a dataset to start clustering and visualization.
                </Alert>
              ) : isRunning ? (
                <Box sx={{ 
                  display: 'flex', 
                  flexDirection: 'column', 
                  alignItems: 'center', 
                  justifyContent: 'center',
                  flex: 1,
                  gap: 2
                }}>
                  <CircularProgress size={60} />
                  <Typography variant="h6">
                    Running {currentAlgorithm.toUpperCase()} Clustering...
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    This may take a few moments depending on your dataset size and parameters.
                  </Typography>
                </Box>
              ) : visualizationData ? (
                <Box sx={{ flex: 1 }}>
                  <Cluster2D
                    points={visualizationData.points}
                    labels={visualizationData.labels}
                    title={`${currentAlgorithm.toUpperCase()} Clustering Results`}
                    xLabel="Component 1"
                    yLabel="Component 2"
                    height={500}
                    onPointClick={(point) => {
                      console.log('Point clicked:', point);
                    }}
                  />
                </Box>
              ) : (
                <Box sx={{ 
                  display: 'flex', 
                  flexDirection: 'column', 
                  alignItems: 'center', 
                  justifyContent: 'center',
                  flex: 1,
                  gap: 2
                }}>
                  <Typography variant="h6" color="text.secondary">
                    No Clustering Results Yet
                  </Typography>
                  <Typography variant="body2" color="text.secondary" textAlign="center">
                    Run a clustering algorithm to see the 2D visualization of your data.
                    The results will appear here with points colored by cluster assignments.
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Box>
      </Box>

      {/* Status Information */}
      {activeRun && (
        <Card sx={{ mt: 2 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Current Run Status
            </Typography>
            <Stack direction="row" spacing={3}>
              <Typography variant="body2">
                <strong>Run ID:</strong> {activeRun.run_id.slice(0, 8)}...
              </Typography>
              <Typography variant="body2">
                <strong>Algorithm:</strong> {activeRun.algorithm.toUpperCase()}
              </Typography>
              <Typography variant="body2">
                <strong>Status:</strong> {activeRun.status}
              </Typography>
              {activeRun.metrics?.silhouette_score && (
                <Typography variant="body2">
                  <strong>Silhouette Score:</strong> {activeRun.metrics.silhouette_score.toFixed(3)}
                </Typography>
              )}
            </Stack>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default VisualizePage;
