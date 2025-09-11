/**
 * AlgorithmConfigPanel component for Interactive Spectral Clustering Platform.
 * Provides algorithm selection tiles and collapsible parameter panels.
 */

import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Chip,
  Collapse,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Stack,
  Paper,
  IconButton,
  Divider,
} from '@mui/material';
import Grid from '@mui/material/Grid';
import {
  ExpandMore,
  ExpandLess,
  TuneOutlined,
  ScatterPlotOutlined,
  AccountTreeOutlined,
  BubbleChartOutlined,
  DonutSmallOutlined,
} from '@mui/icons-material';

interface AlgorithmConfig {
  id: string;
  name: string;
  subtitle: string;
  complexity: 'Low' | 'Medium' | 'High';
  icon: React.ReactNode;
  selected: boolean;
  parameters: Record<string, any>;
}

interface AlgorithmConfigPanelProps {
  onConfigChange?: (configs: AlgorithmConfig[]) => void;
}

export const AlgorithmConfigPanel: React.FC<AlgorithmConfigPanelProps> = ({ onConfigChange }) => {
  const [algorithms, setAlgorithms] = useState<AlgorithmConfig[]>([
    {
      id: 'kmeans',
      name: 'K-Means',
      subtitle: 'Partition-based clustering',
      complexity: 'Low',
      icon: <ScatterPlotOutlined />,
      selected: false,
      parameters: {
        n_clusters: 3,
        init: 'k-means++',
        max_iter: 300,
      },
    },
    {
      id: 'spectral',
      name: 'Spectral Clustering',
      subtitle: 'Graph-based clustering',
      complexity: 'High',
      icon: <AccountTreeOutlined />,
      selected: false,
      parameters: {
        n_clusters: 3,
        affinity: 'rbf',
        gamma: 1.0,
      },
    },
    {
      id: 'dbscan',
      name: 'DBSCAN',
      subtitle: 'Density-based clustering',
      complexity: 'Medium',
      icon: <BubbleChartOutlined />,
      selected: false,
      parameters: {
        eps: 0.5,
        min_samples: 5,
        metric: 'euclidean',
      },
    },
    {
      id: 'hierarchical',
      name: 'Hierarchical',
      subtitle: 'Tree-based clustering',
      complexity: 'Medium',
      icon: <AccountTreeOutlined />,
      selected: false,
      parameters: {
        n_clusters: 3,
        linkage: 'ward',
        metric: 'euclidean',
      },
    },
    {
      id: 'gmm',
      name: 'Gaussian Mixture',
      subtitle: 'Probabilistic clustering',
      complexity: 'High',
      icon: <DonutSmallOutlined />,
      selected: false,
      parameters: {
        n_components: 3,
        covariance_type: 'full',
        max_iter: 100,
      },
    },
  ]);

  const [expandedPanels, setExpandedPanels] = useState<Set<string>>(new Set());

  const handleAlgorithmToggle = (algorithmId: string) => {
    const updatedAlgorithms = algorithms.map((alg) =>
      alg.id === algorithmId ? { ...alg, selected: !alg.selected } : alg,
    );
    setAlgorithms(updatedAlgorithms);
    onConfigChange?.(updatedAlgorithms);
  };

  const handleParameterChange = (algorithmId: string, paramName: string, value: any) => {
    const updatedAlgorithms = algorithms.map((alg) =>
      alg.id === algorithmId
        ? { ...alg, parameters: { ...alg.parameters, [paramName]: value } }
        : alg,
    );
    setAlgorithms(updatedAlgorithms);
    onConfigChange?.(updatedAlgorithms);
  };

  const handlePanelToggle = (algorithmId: string) => {
    const newExpanded = new Set(expandedPanels);
    if (newExpanded.has(algorithmId)) {
      newExpanded.delete(algorithmId);
    } else {
      newExpanded.add(algorithmId);
    }
    setExpandedPanels(newExpanded);
  };

  const getComplexityColor = (complexity: string) => {
    switch (complexity) {
      case 'Low':
        return 'success';
      case 'Medium':
        return 'warning';
      case 'High':
        return 'error';
      default:
        return 'default';
    }
  };

  const renderParameterControls = (algorithm: AlgorithmConfig) => {
    const { id, parameters } = algorithm;

    switch (id) {
      case 'kmeans':
        return (
          <Grid container spacing={3}>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="Number of Clusters"
                type="number"
                value={parameters.n_clusters}
                onChange={(e) => handleParameterChange(id, 'n_clusters', parseInt(e.target.value))}
                inputProps={{ min: 2, max: 20 }}
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <FormControl fullWidth>
                <InputLabel>Initialization Method</InputLabel>
                <Select
                  value={parameters.init}
                  label="Initialization Method"
                  onChange={(e) => handleParameterChange(id, 'init', e.target.value)}
                >
                  <MenuItem value="k-means++">K-means++</MenuItem>
                  <MenuItem value="random">Random</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="Max Iterations"
                type="number"
                value={parameters.max_iter}
                onChange={(e) => handleParameterChange(id, 'max_iter', parseInt(e.target.value))}
                inputProps={{ min: 100, max: 1000 }}
              />
            </Grid>
          </Grid>
        );

      case 'spectral':
        return (
          <Grid container spacing={3}>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="Number of Clusters"
                type="number"
                value={parameters.n_clusters}
                onChange={(e) => handleParameterChange(id, 'n_clusters', parseInt(e.target.value))}
                inputProps={{ min: 2, max: 20 }}
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <FormControl fullWidth>
                <InputLabel>Affinity Function</InputLabel>
                <Select
                  value={parameters.affinity}
                  label="Affinity Function"
                  onChange={(e) => handleParameterChange(id, 'affinity', e.target.value)}
                >
                  <MenuItem value="rbf">RBF</MenuItem>
                  <MenuItem value="linear">Linear</MenuItem>
                  <MenuItem value="poly">Polynomial</MenuItem>
                  <MenuItem value="sigmoid">Sigmoid</MenuItem>
                  <MenuItem value="cosine">Cosine</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="Gamma"
                type="number"
                value={parameters.gamma}
                onChange={(e) => handleParameterChange(id, 'gamma', parseFloat(e.target.value))}
                inputProps={{ min: 0.1, max: 10, step: 0.1 }}
              />
            </Grid>
          </Grid>
        );

      case 'dbscan':
        return (
          <Grid container spacing={3}>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="Epsilon"
                type="number"
                value={parameters.eps}
                onChange={(e) => handleParameterChange(id, 'eps', parseFloat(e.target.value))}
                inputProps={{ min: 0.1, max: 2, step: 0.1 }}
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="Min Samples"
                type="number"
                value={parameters.min_samples}
                onChange={(e) => handleParameterChange(id, 'min_samples', parseInt(e.target.value))}
                inputProps={{ min: 2, max: 20 }}
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <FormControl fullWidth>
                <InputLabel>Distance Metric</InputLabel>
                <Select
                  value={parameters.metric}
                  label="Distance Metric"
                  onChange={(e) => handleParameterChange(id, 'metric', e.target.value)}
                >
                  <MenuItem value="euclidean">Euclidean</MenuItem>
                  <MenuItem value="manhattan">Manhattan</MenuItem>
                  <MenuItem value="cosine">Cosine</MenuItem>
                  <MenuItem value="chebyshev">Chebyshev</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        );

      case 'hierarchical':
        return (
          <Grid container spacing={3}>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="Number of Clusters"
                type="number"
                value={parameters.n_clusters}
                onChange={(e) => handleParameterChange(id, 'n_clusters', parseInt(e.target.value))}
                inputProps={{ min: 2, max: 20 }}
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <FormControl fullWidth>
                <InputLabel>Linkage Method</InputLabel>
                <Select
                  value={parameters.linkage}
                  label="Linkage Method"
                  onChange={(e) => handleParameterChange(id, 'linkage', e.target.value)}
                >
                  <MenuItem value="ward">Ward</MenuItem>
                  <MenuItem value="complete">Complete</MenuItem>
                  <MenuItem value="average">Average</MenuItem>
                  <MenuItem value="single">Single</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={4}>
              <FormControl fullWidth>
                <InputLabel>Distance Metric</InputLabel>
                <Select
                  value={parameters.metric}
                  label="Distance Metric"
                  onChange={(e) => handleParameterChange(id, 'metric', e.target.value)}
                >
                  <MenuItem value="euclidean">Euclidean</MenuItem>
                  <MenuItem value="manhattan">Manhattan</MenuItem>
                  <MenuItem value="cosine">Cosine</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        );

      case 'gmm':
        return (
          <Grid container spacing={3}>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="Number of Components"
                type="number"
                value={parameters.n_components}
                onChange={(e) => handleParameterChange(id, 'n_components', parseInt(e.target.value))}
                inputProps={{ min: 1, max: 20 }}
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <FormControl fullWidth>
                <InputLabel>Covariance Type</InputLabel>
                <Select
                  value={parameters.covariance_type}
                  label="Covariance Type"
                  onChange={(e) => handleParameterChange(id, 'covariance_type', e.target.value)}
                >
                  <MenuItem value="full">Full</MenuItem>
                  <MenuItem value="tied">Tied</MenuItem>
                  <MenuItem value="diag">Diagonal</MenuItem>
                  <MenuItem value="spherical">Spherical</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="Max Iterations"
                type="number"
                value={parameters.max_iter}
                onChange={(e) => handleParameterChange(id, 'max_iter', parseInt(e.target.value))}
                inputProps={{ min: 10, max: 1000 }}
              />
            </Grid>
          </Grid>
        );

      default:
        return (
          <Typography variant="body2" color="text.secondary">
            Parameters for {algorithm.name} will be available soon.
          </Typography>
        );
    }
  };

  return (
    <Card>
      <CardContent>
        <Box display="flex" alignItems="center" gap={2} mb={3}>
          <TuneOutlined color="primary" fontSize="large" />
          <Typography variant="h6">Hyperparameter Configuration</Typography>
        </Box>

        {/* Algorithm Selection Tiles */}
        <Typography variant="subtitle1" gutterBottom>
          Algorithm Selection
        </Typography>
        <Typography variant="body2" color="text.secondary" paragraph>
          Select one or more clustering algorithms to compare. Each algorithm has different
          strengths and complexity levels.
        </Typography>

        <Grid container spacing={2} sx={{ mb: 4 }}>
          {algorithms.map((algorithm) => (
            <Grid item xs={12} sm={6} md={4} key={algorithm.id}>
              <Paper
                elevation={algorithm.selected ? 3 : 1}
                sx={{
                  p: 2,
                  cursor: 'pointer',
                  border: algorithm.selected ? 2 : 1,
                  borderColor: algorithm.selected ? 'primary.main' : 'divider',
                  backgroundColor: algorithm.selected ? 'primary.50' : 'background.paper',
                  transition: 'all 0.2s ease-in-out',
                  '&:hover': {
                    elevation: 2,
                    borderColor: 'primary.main',
                  },
                }}
                onClick={() => handleAlgorithmToggle(algorithm.id)}
              >
                <Stack spacing={2}>
                  <Box display="flex" alignItems="center" justifyContent="space-between">
                    <Box display="flex" alignItems="center" gap={1}>
                      {algorithm.icon}
                      <Typography variant="h6">{algorithm.name}</Typography>
                    </Box>
                    <Chip
                      label={algorithm.complexity}
                      size="small"
                      color={getComplexityColor(algorithm.complexity) as any}
                      variant="outlined"
                    />
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    {algorithm.subtitle}
                  </Typography>
                </Stack>
              </Paper>
            </Grid>
          ))}
        </Grid>

        {/* Parameter Panels */}
        {algorithms.filter((alg) => alg.selected).length > 0 && (
          <>
            <Divider sx={{ my: 3 }} />
            <Typography variant="subtitle1" gutterBottom>
              Algorithm Parameters
            </Typography>

            <Stack spacing={2}>
              {algorithms
                .filter((alg) => alg.selected)
                .map((algorithm) => (
                  <Card key={algorithm.id} variant="outlined">
                    <CardContent>
                      <Box
                        display="flex"
                        alignItems="center"
                        justifyContent="space-between"
                        sx={{ cursor: 'pointer' }}
                        onClick={() => handlePanelToggle(algorithm.id)}
                      >
                        <Box display="flex" alignItems="center" gap={2}>
                          {algorithm.icon}
                          <Typography variant="h6">{algorithm.name} Parameters</Typography>
                        </Box>
                        <IconButton size="small">
                          {expandedPanels.has(algorithm.id) ? <ExpandLess /> : <ExpandMore />}
                        </IconButton>
                      </Box>

                      <Collapse in={expandedPanels.has(algorithm.id)}>
                        <Box sx={{ mt: 3 }}>{renderParameterControls(algorithm)}</Box>
                      </Collapse>
                    </CardContent>
                  </Card>
                ))}
            </Stack>
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default AlgorithmConfigPanel;
