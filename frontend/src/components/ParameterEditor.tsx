import React, { useState, useEffect, useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Tabs,
  Tab,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Switch,
  FormControlLabel,
  Chip,
  Alert,
  Tooltip,
  IconButton,
  Collapse,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Info as InfoIcon,
  Restore as RestoreIcon,
} from '@mui/icons-material';

interface ParameterEditorProps {
  onParametersChange: (algorithm: string, params: Record<string, any>) => void;
}

interface AlgorithmConfig {
  name: string;
  displayName: string;
  description: string;
  parameters: ParameterConfig[];
}

interface ParameterConfig {
  name: string;
  displayName: string;
  type: 'number' | 'select' | 'boolean' | 'slider';
  default: any;
  min?: number;
  max?: number;
  step?: number;
  options?: { value: any; label: string }[];
  description: string;
  advanced?: boolean;
}

const algorithmConfigs: Record<string, AlgorithmConfig> = {
  spectral: {
    name: 'spectral',
    displayName: 'Spectral Clustering',
    description: 'Graph-based clustering using eigenvalue decomposition',
    parameters: [
      {
        name: 'n_clusters',
        displayName: 'Number of Clusters',
        type: 'slider',
        default: 3,
        min: 2,
        max: 20,
        step: 1,
        description: 'Number of clusters to form'
      },
      {
        name: 'gamma',
        displayName: 'RBF Gamma',
        type: 'number',
        default: 1.0,
        min: 0.001,
        max: 100,
        description: 'Kernel coefficient for RBF, poly, sigmoid kernels'
      },
      {
        name: 'affinity',
        displayName: 'Affinity',
        type: 'select',
        default: 'rbf',
        options: [
          { value: 'rbf', label: 'RBF' },
          { value: 'linear', label: 'Linear' },
          { value: 'poly', label: 'Polynomial' },
          { value: 'sigmoid', label: 'Sigmoid' },
          { value: 'cosine', label: 'Cosine' }
        ],
        description: 'How to construct the affinity matrix'
      },
      {
        name: 'n_neighbors',
        displayName: 'k-NN Neighbors',
        type: 'slider',
        default: 10,
        min: 1,
        max: 50,
        step: 1,
        description: 'Number of neighbors for k-NN graph construction',
        advanced: true
      },
      {
        name: 'eigen_solver',
        displayName: 'Eigenvalue Solver',
        type: 'select',
        default: 'arpack',
        options: [
          { value: 'arpack', label: 'ARPACK' },
          { value: 'lobpcg', label: 'LOBPCG' },
          { value: 'amg', label: 'AMG' }
        ],
        description: 'Eigenvalue decomposition strategy',
        advanced: true
      }
    ]
  },
  dbscan: {
    name: 'dbscan',
    displayName: 'DBSCAN',
    description: 'Density-based spatial clustering of applications with noise',
    parameters: [
      {
        name: 'eps',
        displayName: 'Epsilon (ε)',
        type: 'number',
        default: 0.5,
        min: 0.001,
        max: 10,
        description: 'Maximum distance between two samples for one to be considered in the neighborhood of the other'
      },
      {
        name: 'min_samples',
        displayName: 'Min Samples',
        type: 'slider',
        default: 5,
        min: 1,
        max: 50,
        step: 1,
        description: 'Number of samples in a neighborhood for a point to be considered as a core point'
      },
      {
        name: 'metric',
        displayName: 'Distance Metric',
        type: 'select',
        default: 'euclidean',
        options: [
          { value: 'euclidean', label: 'Euclidean' },
          { value: 'manhattan', label: 'Manhattan' },
          { value: 'cosine', label: 'Cosine' },
          { value: 'chebyshev', label: 'Chebyshev' }
        ],
        description: 'Distance metric to use'
      },
      {
        name: 'leaf_size',
        displayName: 'Leaf Size',
        type: 'slider',
        default: 30,
        min: 10,
        max: 100,
        step: 5,
        description: 'Leaf size passed to BallTree or cKDTree',
        advanced: true
      }
    ]
  },
  gmm: {
    name: 'gmm',
    displayName: 'Gaussian Mixture Model',
    description: 'Probabilistic clustering using Gaussian components',
    parameters: [
      {
        name: 'n_components',
        displayName: 'Number of Components',
        type: 'slider',
        default: 3,
        min: 1,
        max: 20,
        step: 1,
        description: 'Number of mixture components'
      },
      {
        name: 'covariance_type',
        displayName: 'Covariance Type',
        type: 'select',
        default: 'full',
        options: [
          { value: 'full', label: 'Full' },
          { value: 'tied', label: 'Tied' },
          { value: 'diag', label: 'Diagonal' },
          { value: 'spherical', label: 'Spherical' }
        ],
        description: 'Type of covariance parameters'
      },
      {
        name: 'max_iter',
        displayName: 'Max Iterations',
        type: 'slider',
        default: 100,
        min: 10,
        max: 1000,
        step: 10,
        description: 'Maximum number of EM iterations',
        advanced: true
      },
      {
        name: 'tol',
        displayName: 'Tolerance',
        type: 'number',
        default: 1e-3,
        min: 1e-6,
        max: 1e-1,
        description: 'Convergence threshold',
        advanced: true
      },
      {
        name: 'init_params',
        displayName: 'Initialization',
        type: 'select',
        default: 'kmeans',
        options: [
          { value: 'kmeans', label: 'K-means' },
          { value: 'random', label: 'Random' }
        ],
        description: 'Method to initialize the weights',
        advanced: true
      }
    ]
  },
  agglomerative: {
    name: 'agglomerative',
    displayName: 'Agglomerative Clustering',
    description: 'Hierarchical clustering using bottom-up approach',
    parameters: [
      {
        name: 'n_clusters',
        displayName: 'Number of Clusters',
        type: 'slider',
        default: 3,
        min: 2,
        max: 20,
        step: 1,
        description: 'Number of clusters to find'
      },
      {
        name: 'linkage',
        displayName: 'Linkage Criterion',
        type: 'select',
        default: 'ward',
        options: [
          { value: 'ward', label: 'Ward' },
          { value: 'complete', label: 'Complete' },
          { value: 'average', label: 'Average' },
          { value: 'single', label: 'Single' }
        ],
        description: 'Linkage criterion to use'
      },
      {
        name: 'metric',
        displayName: 'Distance Metric',
        type: 'select',
        default: 'euclidean',
        options: [
          { value: 'euclidean', label: 'Euclidean' },
          { value: 'manhattan', label: 'Manhattan' },
          { value: 'cosine', label: 'Cosine' }
        ],
        description: 'Distance metric to use'
      },
      {
        name: 'distance_threshold',
        displayName: 'Distance Threshold',
        type: 'number',
        default: null,
        min: 0,
        max: 100,
        description: 'Linkage distance threshold above which clusters will not be merged (set to enable)',
        advanced: true
      }
    ]
  }
};

export const ParameterEditor: React.FC<ParameterEditorProps> = ({
  onParametersChange,
}) => {
  const [activeTab, setActiveTab] = useState(0);
  const [parameters, setParameters] = useState<Record<string, Record<string, any>>>({});
  const [showAdvanced, setShowAdvanced] = useState<Record<string, boolean>>({});
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({});

  const algorithms = useMemo(() => Object.keys(algorithmConfigs), []);

  // Initialize parameters with defaults
  useEffect(() => {
    const initialParams: Record<string, Record<string, any>> = {};
    algorithms.forEach(alg => {
      initialParams[alg] = {};
      algorithmConfigs[alg].parameters.forEach(param => {
        initialParams[alg][param.name] = param.default;
      });
    });
    setParameters(initialParams);
  }, [algorithms]);

  // Notify parent when parameters change
  useEffect(() => {
    algorithms.forEach(alg => {
      if (parameters[alg]) {
        onParametersChange(alg, parameters[alg]);
      }
    });
  }, [parameters, algorithms]); // Removed onParametersChange to prevent infinite loops

  const handleParameterChange = (algorithm: string, paramName: string, value: any) => {
    setParameters(prev => ({
      ...prev,
      [algorithm]: {
        ...prev[algorithm],
        [paramName]: value
      }
    }));
  };

  const resetToDefaults = (algorithm: string) => {
    const defaults: Record<string, any> = {};
    algorithmConfigs[algorithm].parameters.forEach(param => {
      defaults[param.name] = param.default;
    });
    setParameters(prev => ({
      ...prev,
      [algorithm]: defaults
    }));
  };

  const toggleAdvanced = (algorithm: string) => {
    setShowAdvanced(prev => ({
      ...prev,
      [algorithm]: !prev[algorithm]
    }));
  };

  const toggleSection = (algorithm: string) => {
    setExpandedSections(prev => ({
      ...prev,
      [algorithm]: !prev[algorithm]
    }));
  };

  const renderParameterInput = (algorithm: string, param: ParameterConfig) => {
    const value = parameters[algorithm]?.[param.name] ?? param.default;

    switch (param.type) {
      case 'slider':
        return (
          <Box>
            <Typography variant="body2" gutterBottom>
              {param.displayName}: {value}
            </Typography>
            <Slider
              value={value}
              onChange={(_, newValue) => handleParameterChange(algorithm, param.name, newValue)}
              min={param.min}
              max={param.max}
              step={param.step}
              marks
              valueLabelDisplay="auto"
            />
          </Box>
        );

      case 'select':
        return (
          <FormControl fullWidth size="small">
            <InputLabel>{param.displayName}</InputLabel>
            <Select
              value={value}
              label={param.displayName}
              onChange={(e) => handleParameterChange(algorithm, param.name, e.target.value)}
            >
              {param.options?.map(option => (
                <MenuItem key={option.value} value={option.value}>
                  {option.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        );

      case 'boolean':
        return (
          <FormControlLabel
            control={
              <Switch
                checked={value}
                onChange={(e) => handleParameterChange(algorithm, param.name, e.target.checked)}
              />
            }
            label={param.displayName}
          />
        );

      case 'number':
      default:
        return (
          <TextField
            label={param.displayName}
            type="number"
            value={value ?? ''}
            onChange={(e) => {
              const newValue = e.target.value === '' ? null : parseFloat(e.target.value);
              handleParameterChange(algorithm, param.name, newValue);
            }}
            inputProps={{
              min: param.min,
              max: param.max,
              step: param.step || 0.01
            }}
            size="small"
            fullWidth
          />
        );
    }
  };

  const renderAlgorithmParameters = (algorithm: string) => {
    const config = algorithmConfigs[algorithm];
    const basicParams = config.parameters.filter(p => !p.advanced);
    const advancedParams = config.parameters.filter(p => p.advanced);

    return (
      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
            <Box display="flex" alignItems="center" gap={1}>
              <Typography variant="h6">
                {config.displayName}
              </Typography>
              <Tooltip title={config.description}>
                <IconButton size="small">
                  <InfoIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
            <Box display="flex" alignItems="center" gap={1}>
              <Tooltip title="Reset to defaults">
                <IconButton 
                  size="small" 
                  onClick={() => resetToDefaults(algorithm)}
                >
                  <RestoreIcon fontSize="small" />
                </IconButton>
              </Tooltip>
              <IconButton
                size="small"
                onClick={() => toggleSection(algorithm)}
                sx={{
                  transform: expandedSections[algorithm] ? 'rotate(180deg)' : 'rotate(0deg)',
                  transition: 'transform 0.2s'
                }}
              >
                <ExpandMoreIcon />
              </IconButton>
            </Box>
          </Box>

          <Collapse in={expandedSections[algorithm] !== false}>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              {config.description}
            </Typography>

            <Box display="flex" flexDirection="column" gap={3}>
              {/* Basic Parameters */}
              {basicParams.map(param => (
                <Box key={param.name}>
                  <Box display="flex" alignItems="center" gap={1} mb={1}>
                    <Typography variant="body2" fontWeight="medium">
                      {param.displayName}
                    </Typography>
                    <Tooltip title={param.description}>
                      <InfoIcon fontSize="small" color="action" />
                    </Tooltip>
                  </Box>
                  {renderParameterInput(algorithm, param)}
                </Box>
              ))}

              {/* Advanced Parameters */}
              {advancedParams.length > 0 && (
                <Box>
                  <Box display="flex" alignItems="center" gap={1} mb={2}>
                    <Typography variant="body2" fontWeight="medium">
                      Advanced Parameters
                    </Typography>
                    <Switch
                      size="small"
                      checked={showAdvanced[algorithm] || false}
                      onChange={() => toggleAdvanced(algorithm)}
                    />
                  </Box>

                  <Collapse in={showAdvanced[algorithm] || false}>
                    <Box display="flex" flexDirection="column" gap={3}>
                      {advancedParams.map(param => (
                        <Box key={param.name}>
                          <Box display="flex" alignItems="center" gap={1} mb={1}>
                            <Typography variant="body2" fontWeight="medium">
                              {param.displayName}
                            </Typography>
                            <Tooltip title={param.description}>
                              <InfoIcon fontSize="small" color="action" />
                            </Tooltip>
                          </Box>
                          {renderParameterInput(algorithm, param)}
                        </Box>
                      ))}
                    </Box>
                  </Collapse>
                </Box>
              )}
            </Box>
          </Collapse>
        </CardContent>
      </Card>
    );
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Algorithm Parameters
      </Typography>
      
      <Alert severity="info" sx={{ mb: 3 }}>
        Configure parameters for each clustering algorithm. Advanced parameters are hidden by default.
        Hover over the info icons for parameter descriptions.
      </Alert>

      <Tabs 
        value={activeTab} 
        onChange={(_, newValue) => setActiveTab(newValue)}
        variant="scrollable"
        scrollButtons="auto"
        sx={{ mb: 3 }}
      >
        <Tab label="All Algorithms" />
        {algorithms.map((alg, index) => (
          <Tab 
            key={alg} 
            label={algorithmConfigs[alg].displayName}
            icon={
              <Chip 
                label={Object.keys(parameters[alg] || {}).length} 
                size="small" 
                color="primary"
              />
            }
            iconPosition="end"
          />
        ))}
      </Tabs>

      <Box>
        {activeTab === 0 ? (
          // Show all algorithms
          algorithms.map(alg => (
            <Box key={alg}>
              {renderAlgorithmParameters(alg)}
            </Box>
          ))
        ) : (
          // Show specific algorithm
          <Box>
            {renderAlgorithmParameters(algorithms[activeTab - 1])}
          </Box>
        )}
      </Box>
    </Box>
  );
};
