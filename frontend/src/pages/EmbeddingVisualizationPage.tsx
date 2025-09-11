import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Container,
  Typography,
  Paper,
  Alert,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import { EmbeddingControls } from '../features/visualize/EmbeddingControls';
import { Cluster2D } from '../features/visualize/Cluster2D';

interface EmbeddingPoint {
  x: number;
  y: number;
  original_index: number;
}

interface EmbeddingResponse {
  dataset_id: string;
  method: string;
  parameters: Record<string, any>;
  points: EmbeddingPoint[];
  execution_time: number;
  cache_hit: boolean;
  explained_variance_ratio?: number[];
  kl_divergence?: number;
}

interface Dataset {
  id: string;
  name: string;
  shape: [number, number];
}

interface ClusterRun {
  id: string;
  dataset_id: string;
  method: string;
  labels: number[];
}

export const EmbeddingVisualizationPage: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [clusterRuns, setClusterRuns] = useState<ClusterRun[]>([]);
  const [selectedRun, setSelectedRun] = useState<string>('');
  const [embeddings, setEmbeddings] = useState<EmbeddingPoint[] | null>(null);
  const [embeddingMethod, setEmbeddingMethod] = useState<string>('pca');
  const [embeddingParams, setEmbeddingParams] = useState<Record<string, any>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch available datasets
  useEffect(() => {
    fetchDatasets();
  }, []);

  // Fetch cluster runs when dataset changes
  useEffect(() => {
    if (selectedDataset) {
      fetchClusterRuns(selectedDataset);
    }
  }, [selectedDataset]);

  const fetchDatasets = async () => {
    try {
      // Mock data for now - replace with actual API call
      const mockDatasets: Dataset[] = [
        { id: 'dataset_1', name: 'Iris Dataset', shape: [150, 4] },
        { id: 'dataset_2', name: 'Wine Dataset', shape: [178, 13] },
        { id: 'dataset_3', name: 'Breast Cancer Dataset', shape: [569, 30] },
      ];
      setDatasets(mockDatasets);
      
      if (mockDatasets.length > 0) {
        setSelectedDataset(mockDatasets[0].id);
      }
    } catch (error) {
      console.error('Error fetching datasets:', error);
      setError('Failed to load datasets');
    }
  };

  const fetchClusterRuns = async (datasetId: string) => {
    try {
      // Mock data for now - replace with actual API call
      const mockRuns: ClusterRun[] = [
        {
          id: 'run_1',
          dataset_id: datasetId,
          method: 'spectral',
          labels: Array.from({ length: 150 }, (_, i) => Math.floor(i / 50))
        },
        {
          id: 'run_2',
          dataset_id: datasetId,
          method: 'dbscan',
          labels: Array.from({ length: 150 }, (_, i) => Math.floor(Math.random() * 3))
        },
      ];
      setClusterRuns(mockRuns);
      
      if (mockRuns.length > 0) {
        setSelectedRun(mockRuns[0].id);
      }
    } catch (error) {
      console.error('Error fetching cluster runs:', error);
      setError('Failed to load cluster runs');
    }
  };

  const generateEmbedding = useCallback(async (method: string, parameters: Record<string, any>) => {
    if (!selectedDataset) {
      setError('Please select a dataset first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const request = {
        dataset_id: selectedDataset,
        method,
        parameters
      };

      const response = await fetch('/api/embed', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result: EmbeddingResponse = await response.json();
      setEmbeddings(result.points);
      setEmbeddingMethod(method);
      setEmbeddingParams(parameters);

      // Show performance info
      console.log(`Generated ${method} embedding in ${result.execution_time.toFixed(3)}s (cache: ${result.cache_hit})`);
      
    } catch (error) {
      console.error('Error generating embedding:', error);
      setError(error instanceof Error ? error.message : 'Failed to generate embedding');
      
      // Generate mock embedding for development
      const mockPoints: EmbeddingPoint[] = Array.from({ length: 150 }, (_, i) => ({
        x: Math.random() * 10 - 5,
        y: Math.random() * 10 - 5,
        original_index: i
      }));
      setEmbeddings(mockPoints);
      setEmbeddingMethod(method);
      setEmbeddingParams(parameters);
      
    } finally {
      setLoading(false);
    }
  }, [selectedDataset]);

  const handleEmbeddingChange = (method: string, parameters: Record<string, any>) => {
    generateEmbedding(method, parameters);
  };

  const generateInitialEmbedding = () => {
    generateEmbedding('pca', { random_state: 42 });
  };

  const getSelectedClusterRun = () => {
    return clusterRuns.find(run => run.id === selectedRun);
  };

  const selectedRunData = getSelectedClusterRun();

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <Typography variant="h4" gutterBottom>
        Dimensionality Reduction Visualization
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Generate 2D embeddings using PCA, t-SNE, or UMAP and visualize clustering results
      </Typography>

      {/* Dataset and Run Selection */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Data Selection
        </Typography>
        
        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr 1fr' }, gap: 2 }}>
          <FormControl fullWidth>
            <InputLabel>Dataset</InputLabel>
            <Select
              value={selectedDataset}
              label="Dataset"
              onChange={(e) => setSelectedDataset(e.target.value)}
            >
              {datasets.map((dataset) => (
                <MenuItem key={dataset.id} value={dataset.id}>
                  {dataset.name} ({dataset.shape[0]} × {dataset.shape[1]})
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <FormControl fullWidth>
            <InputLabel>Cluster Run</InputLabel>
            <Select
              value={selectedRun}
              label="Cluster Run"
              onChange={(e) => setSelectedRun(e.target.value)}
              disabled={clusterRuns.length === 0}
            >
              {clusterRuns.map((run) => (
                <MenuItem key={run.id} value={run.id}>
                  {run.method} (Run {run.id})
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <Button
            variant="contained"
            onClick={generateInitialEmbedding}
            disabled={!selectedDataset || loading}
            fullWidth
          >
            Generate Embedding
          </Button>
        </Box>
      </Paper>

      {/* Error Display */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Main Content */}
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: '350px 1fr' }, gap: 3 }}>
        {/* Controls Panel */}
        <Box>
          <EmbeddingControls
            onEmbeddingChange={handleEmbeddingChange}
            disabled={!selectedDataset}
            loading={loading}
          />
        </Box>

        {/* Visualization Panel */}
        <Box>
          <Cluster2D
            embeddings={embeddings || undefined}
            labels={selectedRunData?.labels}
            embeddingMethod={embeddingMethod}
            width={700}
            height={500}
            onPointClick={(point) => {
              console.log('Clicked point:', point);
            }}
          />
        </Box>
      </Box>

      {/* Embedding Info */}
      {embeddings && !loading && (
        <Paper sx={{ p: 2, mt: 3 }}>
          <Typography variant="subtitle2" gutterBottom>
            Embedding Information
          </Typography>
          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'repeat(3, 1fr)' }, gap: 2 }}>
            <Box>
              <Typography variant="caption" color="text.secondary">Method</Typography>
              <Typography variant="body2">{embeddingMethod.toUpperCase()}</Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">Points</Typography>
              <Typography variant="body2">{embeddings.length}</Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">Parameters</Typography>
              <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>
                {JSON.stringify(embeddingParams, null, 2)}
              </Typography>
            </Box>
          </Box>
        </Paper>
      )}
    </Container>
  );
};
