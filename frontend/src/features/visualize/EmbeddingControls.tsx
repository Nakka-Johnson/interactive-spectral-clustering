import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  TextField,
  Button,
  Collapse,
  IconButton,
  Chip,
  Alert,
  CircularProgress,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Refresh as RefreshIcon,
  Info as InfoIcon,
} from '@mui/icons-material';

interface EmbeddingMethod {
  value: string;
  label: string;
  description: string;
  speed: string;
  preserves: string;
  best_for: string;
}

interface EmbeddingParameters {
  random_state?: number;
  svd_solver?: string;
  perplexity?: number;
  learning_rate?: number;
  n_iter?: number;
  init?: string;
  n_neighbors?: number;
  min_dist?: number;
}

interface EmbeddingControlsProps {
  onEmbeddingChange: (method: string, parameters: EmbeddingParameters) => void;
  disabled?: boolean;
  loading?: boolean;
}

const EMBEDDING_METHODS: EmbeddingMethod[] = [
  {
    value: 'pca',
    label: 'PCA',
    description: 'Principal Component Analysis - linear dimensionality reduction',
    speed: 'fast',
    preserves: 'global_structure',
    best_for: 'Linear relationships, global structure preservation'
  },
  {
    value: 'tsne',
    label: 't-SNE',
    description: 't-Distributed Stochastic Neighbor Embedding - nonlinear reduction',
    speed: 'slow',
    preserves: 'local_structure',
    best_for: 'Local neighborhoods, cluster visualization'
  },
  {
    value: 'umap',
    label: 'UMAP',
    description: 'Uniform Manifold Approximation and Projection - fast nonlinear reduction',
    speed: 'medium',
    preserves: 'both_local_and_global',
    best_for: 'Balance of local and global structure'
  }
];

const DEFAULT_PARAMETERS: Record<string, EmbeddingParameters> = {
  pca: {
    random_state: 42,
    svd_solver: 'auto'
  },
  tsne: {
    perplexity: 30,
    learning_rate: 200,
    n_iter: 1000,
    random_state: 42,
    init: 'random'
  },
  umap: {
    n_neighbors: 15,
    min_dist: 0.1,
    random_state: 42
  }
};

export const EmbeddingControls: React.FC<EmbeddingControlsProps> = ({
  onEmbeddingChange,
  disabled = false,
  loading = false,
}) => {
  const [selectedMethod, setSelectedMethod] = useState<string>('pca');
  const [parameters, setParameters] = useState<Record<string, any>>(DEFAULT_PARAMETERS.pca);
  const [expanded, setExpanded] = useState(false);
  const [availableMethods, setAvailableMethods] = useState<string[]>(['pca', 'tsne']);

  // Check available methods on mount
  useEffect(() => {
    checkAvailableMethods();
  }, []);

  const checkAvailableMethods = async () => {
    try {
      const response = await fetch('/api/embed/methods');
      if (response.ok) {
        const data = await response.json();
        setAvailableMethods(data.methods || ['pca', 'tsne']);
      }
    } catch (error) {
       console.error('Error fetching methods:', error);
    }
  };

  const handleMethodChange = (method: string) => {
    setSelectedMethod(method);
    const newParams = { ...DEFAULT_PARAMETERS[method as keyof typeof DEFAULT_PARAMETERS] };
    setParameters(newParams);
    onEmbeddingChange(method, newParams);
  };

  const handleParameterChange = (param: string, value: any) => {
    const newParams = { ...parameters, [param]: value };
    setParameters(newParams);
    onEmbeddingChange(selectedMethod, newParams);
  };

  const resetToDefaults = () => {
    const defaultParams = { ...DEFAULT_PARAMETERS[selectedMethod as keyof typeof DEFAULT_PARAMETERS] };
    setParameters(defaultParams);
    onEmbeddingChange(selectedMethod, defaultParams);
  };

  const getMethodInfo = (methodValue: string): EmbeddingMethod => {
    return EMBEDDING_METHODS.find(m => m.value === methodValue) || EMBEDDING_METHODS[0];
  };

  const getSpeedColor = (speed: string) => {
    switch (speed) {
      case 'fast': return 'success';
      case 'medium': return 'warning';
      case 'slow': return 'error';
      default: return 'default';
    }
  };

  const renderParameterControls = () => {
    const method = selectedMethod;
    
    switch (method) {
      case 'pca':
        return (
          <Box sx={{ mt: 2 }}>
            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 2 }}>
              <FormControl fullWidth size="small">
                <InputLabel>SVD Solver</InputLabel>
                <Select
                  value={parameters.svd_solver || 'auto'}
                  label="SVD Solver"
                  onChange={(e) => handleParameterChange('svd_solver', e.target.value)}
                >
                  <MenuItem value="auto">Auto</MenuItem>
                  <MenuItem value="full">Full</MenuItem>
                  <MenuItem value="arpack">ARPACK</MenuItem>
                  <MenuItem value="randomized">Randomized</MenuItem>
                </Select>
              </FormControl>
              <TextField
                fullWidth
                size="small"
                label="Random State"
                type="number"
                value={parameters.random_state || 42}
                onChange={(e) => handleParameterChange('random_state', parseInt(e.target.value) || 42)}
              />
            </Box>
          </Box>
        );

      case 'tsne':
        return (
          <Box sx={{ mt: 2 }}>
            <Box sx={{ mb: 2 }}>
              <Typography gutterBottom>
                Perplexity: {parameters.perplexity || 30}
              </Typography>
              <Slider
                value={parameters.perplexity || 30}
                onChange={(_, value) => handleParameterChange('perplexity', value)}
                min={5}
                max={50}
                step={1}
                marks={[
                  { value: 5, label: '5' },
                  { value: 30, label: '30' },
                  { value: 50, label: '50' }
                ]}
              />
            </Box>
            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 2 }}>
              <TextField
                fullWidth
                size="small"
                label="Learning Rate"
                type="number"
                value={parameters.learning_rate || 200}
                onChange={(e) => handleParameterChange('learning_rate', parseFloat(e.target.value) || 200)}
              />
              <TextField
                fullWidth
                size="small"
                label="Max Iterations"
                type="number"
                value={parameters.n_iter || 1000}
                onChange={(e) => handleParameterChange('n_iter', parseInt(e.target.value) || 1000)}
              />
            </Box>
          </Box>
        );

      case 'umap':
        return (
          <Box sx={{ mt: 2 }}>
            <Box sx={{ mb: 2 }}>
              <Typography gutterBottom>
                Neighbors: {parameters.n_neighbors || 15}
              </Typography>
              <Slider
                value={parameters.n_neighbors || 15}
                onChange={(_, value) => handleParameterChange('n_neighbors', value)}
                min={2}
                max={200}
                step={1}
                marks={[
                  { value: 2, label: '2' },
                  { value: 15, label: '15' },
                  { value: 200, label: '200' }
                ]}
              />
            </Box>
            <Box sx={{ mb: 2 }}>
              <Typography gutterBottom>
                Min Distance: {parameters.min_dist || 0.1}
              </Typography>
              <Slider
                value={parameters.min_dist || 0.1}
                onChange={(_, value) => handleParameterChange('min_dist', value)}
                min={0.0}
                max={1.0}
                step={0.01}
                marks={[
                  { value: 0.0, label: '0.0' },
                  { value: 0.1, label: '0.1' },
                  { value: 1.0, label: '1.0' }
                ]}
              />
            </Box>
          </Box>
        );

      default:
        return null;
    }
  };

  const currentMethodInfo = getMethodInfo(selectedMethod);

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Typography variant="h6">
            Dimensionality Reduction
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {loading && <CircularProgress size={20} />}
            <IconButton
              onClick={() => setExpanded(!expanded)}
              size="small"
            >
              {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
            </IconButton>
          </Box>
        </Box>

        {/* Method Selection */}
        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '2fr 1fr' }, gap: 2, alignItems: 'center' }}>
          <FormControl fullWidth size="small">
            <InputLabel>Embedding Method</InputLabel>
            <Select
              value={selectedMethod}
              label="Embedding Method"
              onChange={(e) => handleMethodChange(e.target.value)}
              disabled={disabled || loading}
            >
              {EMBEDDING_METHODS.filter(method => 
                availableMethods.includes(method.value)
              ).map((method) => (
                <MenuItem key={method.value} value={method.value}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    {method.label}
                    <Chip
                      label={method.speed}
                      size="small"
                      color={getSpeedColor(method.speed) as any}
                      variant="outlined"
                    />
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <Button
            fullWidth
            variant="outlined"
            size="small"
            startIcon={<RefreshIcon />}
            onClick={resetToDefaults}
            disabled={disabled || loading}
          >
            Reset
          </Button>
        </Box>

        {/* Method Info */}
        <Box sx={{ mt: 2, p: 1, bgcolor: 'background.paper', borderRadius: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
            <InfoIcon fontSize="small" color="primary" />
            <Typography variant="subtitle2">
              {currentMethodInfo.label}
            </Typography>
          </Box>
          <Typography variant="body2" color="text.secondary">
            {currentMethodInfo.description}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            Best for: {currentMethodInfo.best_for}
          </Typography>
        </Box>

        {/* Parameter Controls */}
        <Collapse in={expanded}>
          {renderParameterControls()}
          
          {!availableMethods.includes('umap') && selectedMethod === 'umap' && (
            <Alert severity="warning" sx={{ mt: 2 }}>
              UMAP is not available. Install with: pip install umap-learn
            </Alert>
          )}
        </Collapse>
      </CardContent>
    </Card>
  );
};

export default EmbeddingControls;
