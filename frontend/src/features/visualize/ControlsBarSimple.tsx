/**
 * Simplified ControlsBar component for Interactive Spectral Clustering Platform.
 */

import React, { useState, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  TextField,
  Switch,
  FormControlLabel,
  Stack,
  Alert
} from '@mui/material';
import { PlayArrow, Memory } from '@mui/icons-material';
import { useAppStore, useDatasets, useActiveDataset } from '../../store/appStore';
import type { AlgorithmId, ParameterMap } from '../../types/api';

interface ControlsBarProps {
  onRunClustering?: (config: {
    datasetId: string;
    algorithm: AlgorithmId;
    parameters: ParameterMap;
  }) => void;
  onParametersChange?: (algorithm: AlgorithmId, parameters: ParameterMap) => void;
  disabled?: boolean;
}

export const ControlsBar: React.FC<ControlsBarProps> = ({
  onRunClustering,
  onParametersChange,
  disabled = false
}) => {
  const { setActiveDataset } = useAppStore();
  const datasets = useDatasets();
  const activeDataset = useActiveDataset();
  
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('spectral');
  const [nClusters, setNClusters] = useState(3);
  const [useGPU, setUseGPU] = useState(true);
  const [isRunning, setIsRunning] = useState(false);

  const handleAlgorithmChange = useCallback((algorithm: string) => {
    setSelectedAlgorithm(algorithm);
    const parameters: ParameterMap = {
      n_clusters: nClusters,
      use_gpu: useGPU
    };
    
    if (algorithm === 'dbscan') {
      parameters.eps = 0.5;
      parameters.min_samples = 5;
      delete parameters.n_clusters;
    } else if (algorithm === 'spectral') {
      parameters.sigma = 1.0;
      parameters.n_neighbors = 10;
    }
    
    onParametersChange?.(algorithm as AlgorithmId, parameters);
  }, [nClusters, useGPU, onParametersChange]);

  const handleRun = useCallback(async () => {
    if (!activeDataset) {
      alert('Please select a dataset first');
      return;
    }

    setIsRunning(true);
    
    const parameters: ParameterMap = {
      n_clusters: nClusters,
      use_gpu: useGPU
    };
    
    if (selectedAlgorithm === 'dbscan') {
      parameters.eps = 0.5;
      parameters.min_samples = 5;
      delete parameters.n_clusters;
    } else if (selectedAlgorithm === 'spectral') {
      parameters.sigma = 1.0;
      parameters.n_neighbors = 10;
    }
    
    try {
      if (onRunClustering) {
        await onRunClustering({
          datasetId: activeDataset.job_id,
          algorithm: selectedAlgorithm as AlgorithmId,
          parameters
        });
      }
    } catch (error) {
      console.error('Failed to start clustering:', error);
    } finally {
      setIsRunning(false);
    }
  }, [activeDataset, selectedAlgorithm, nClusters, useGPU, onRunClustering]);

  const canRun = !!activeDataset && !disabled && !isRunning;

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Clustering Configuration
        </Typography>
        
        <Stack spacing={3}>
          {/* Dataset Selection */}
          <FormControl fullWidth>
            <InputLabel>Dataset</InputLabel>
            <Select
              value={activeDataset?.job_id || ''}
              label="Dataset"
              onChange={(e) => {
                const dataset = datasets.find(d => d.job_id === e.target.value);
                setActiveDataset(dataset || null);
              }}
              disabled={disabled}
            >
              {datasets.map((dataset) => (
                <MenuItem key={dataset.job_id} value={dataset.job_id}>
                  <Box>
                    <Typography variant="body2">{dataset.name}</Typography>
                    <Typography variant="caption" color="text.secondary">
                      {dataset.shape[0]} rows × {dataset.shape[1]} columns
                    </Typography>
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {/* Algorithm Selection */}
          <FormControl fullWidth>
            <InputLabel>Algorithm</InputLabel>
            <Select
              value={selectedAlgorithm}
              label="Algorithm"
              onChange={(e) => handleAlgorithmChange(e.target.value)}
              disabled={disabled}
            >
              <MenuItem value="spectral">Spectral Clustering</MenuItem>
              <MenuItem value="kmeans">K-Means</MenuItem>
              <MenuItem value="dbscan">DBSCAN</MenuItem>
              <MenuItem value="gmm">Gaussian Mixture Model</MenuItem>
              <MenuItem value="agglomerative">Agglomerative Clustering</MenuItem>
            </Select>
          </FormControl>

          {/* Number of Clusters (not for DBSCAN) */}
          {selectedAlgorithm !== 'dbscan' && (
            <TextField
              fullWidth
              label="Number of Clusters"
              type="number"
              value={nClusters}
              onChange={(e) => setNClusters(parseInt(e.target.value) || 3)}
              inputProps={{ min: 2, max: 20 }}
              disabled={disabled}
            />
          )}

          {/* GPU Toggle */}
          <FormControlLabel
            control={
              <Switch
                checked={useGPU}
                onChange={(e) => setUseGPU(e.target.checked)}
                disabled={disabled}
              />
            }
            label={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Memory />
                <Typography>Use GPU Acceleration</Typography>
              </Box>
            }
          />

          {/* Validation */}
          {!activeDataset && (
            <Alert severity="error">
              Please select a dataset first
            </Alert>
          )}

          {/* Run Button */}
          <Button
            variant="contained"
            size="large"
            onClick={handleRun}
            disabled={!canRun}
            startIcon={<PlayArrow />}
          >
            {isRunning ? 'Running...' : 'Run Clustering'}
          </Button>
        </Stack>
      </CardContent>
    </Card>
  );
};

export default ControlsBar;
