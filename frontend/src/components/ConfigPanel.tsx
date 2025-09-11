import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  FormControl,
  FormGroup,
  FormControlLabel,
  Checkbox,
  TextField,
  Switch,
  Button,
  Alert,
  Chip,
  Select,
  MenuItem,
  InputLabel,
} from '@mui/material';
import {
  Settings,
  PlayArrow,
  Tune,
} from '@mui/icons-material';
import { useClusteringStore } from '../store';
// import { ParameterEditor } from './ParameterEditor';

const CLUSTERING_METHODS = [
  { id: 'kmeans', label: 'K-Means', description: 'Centroid-based clustering' },
  { id: 'spectral', label: 'Spectral Clustering', description: 'Graph-based clustering' },
  { id: 'manual_spectral', label: 'Manual Spectral', description: 'Custom spectral implementation' },
  { id: 'dbscan', label: 'DBSCAN', description: 'Density-based clustering' },
  { id: 'agglomerative', label: 'Agglomerative', description: 'Hierarchical clustering' },
  { id: 'gmm', label: 'Gaussian Mixture Model', description: 'Probabilistic model-based clustering' },
];

const DIM_REDUCERS = [
  { id: 'pca', label: 'PCA', description: 'Principal Component Analysis' },
  { id: 'tsne', label: 't-SNE', description: 't-Distributed Stochastic Neighbor Embedding' },
  { id: 'umap', label: 'UMAP', description: 'Uniform Manifold Approximation and Projection' },
];

const ConfigPanel: React.FC = () => {
  const {
    dataset,
    parameters,
    setParameters,
    validateParameters,
    canRunClustering,
    setCurrentTab,
  } = useClusteringStore();
  
  // Advanced parameters - currently not used in API
  // const [advancedParameters, setAdvancedParameters] = useState<Record<string, Record<string, any>>>({});

  const validation = validateParameters();

  const handleMethodChange = (methodId: string, checked: boolean) => {
    const newMethods = checked
      ? [...parameters.methods, methodId]
      : parameters.methods.filter(m => m !== methodId);
    
    setParameters({ methods: newMethods });
  };

  const handleParameterChange = (key: keyof typeof parameters, value: any) => {
    setParameters({ [key]: value });
  };

  // const handleAdvancedParametersChange = (algorithm: string, params: Record<string, any>) => {
  //   setAdvancedParameters(prev => ({
  //     ...prev,
  //     [algorithm]: params
  //   }));
  // };

  const handleRunClustering = () => {
    if (canRunClustering()) {
      // Run clustering with current parameters
      // Advanced parameters are handled separately in the API call
      setCurrentTab('results');
    }
  };

  if (!dataset) {
    return (
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="h6" color="text.secondary">
          Upload a dataset to configure clustering parameters
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Configuration
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Configure clustering algorithms and parameters
      </Typography>

      {/* Algorithm Selection */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <Settings sx={{ mr: 1 }} />
            <Typography variant="h6">
              Algorithm Selection
            </Typography>
          </Box>
          
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Select one or more clustering algorithms to compare
          </Typography>

          <FormControl component="fieldset">
            <FormGroup>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                {CLUSTERING_METHODS.map((method) => (
                  <Card variant="outlined" key={method.id}>
                    <CardContent sx={{ p: 2 }}>
                      <FormControlLabel
                        control={
                          <Checkbox
                            checked={parameters.methods.includes(method.id)}
                            onChange={(e) => handleMethodChange(method.id, e.target.checked)}
                            aria-label={`Select ${method.label} clustering algorithm`}
                            tabIndex={0}
                          />
                        }
                        label={method.label}
                        sx={{ mb: 1 }}
                      />
                      <Typography variant="body2" color="text.secondary">
                        {method.description}
                      </Typography>
                    </CardContent>
                  </Card>
                ))}
              </Box>
            </FormGroup>
          </FormControl>

          {parameters.methods.length === 0 && (
            <Alert severity="warning" sx={{ mt: 2 }}>
              Please select at least one clustering method
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Hyperparameters */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Hyperparameters
          </Typography>
          
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            {/* Number of Clusters */}
            <TextField
              fullWidth
              label="Number of Clusters"
              type="number"
              value={parameters.n_clusters}
              onChange={(e) => handleParameterChange('n_clusters', parseInt(e.target.value))}
              inputProps={{ min: 1, max: 20, tabIndex: 0 }}
              aria-label="Set number of clusters to find"
              helperText="Number of clusters to find (for K-means, Spectral, Agglomerative)"
            />

            {/* Sigma */}
            <TextField
              fullWidth
              label="Sigma (RBF Kernel Bandwidth)"
              type="number"
              value={parameters.sigma}
              onChange={(e) => handleParameterChange('sigma', parseFloat(e.target.value))}
              inputProps={{ min: 0.1, max: 10, step: 0.1, tabIndex: 0 }}
              aria-label="Set RBF kernel bandwidth parameter"
              helperText="Bandwidth parameter for RBF kernel in spectral clustering"
            />

            {/* Number of Neighbors */}
            <TextField
              fullWidth
              label="Number of Neighbors"
              type="number"
              value={parameters.n_neighbors}
              onChange={(e) => handleParameterChange('n_neighbors', parseInt(e.target.value))}
              inputProps={{ min: 1, max: 50, tabIndex: 0 }}
              aria-label="Set number of neighbors for k-NN graph"
              helperText="Number of neighbors for k-NN graph construction"
            />
          </Box>
        </CardContent>
      </Card>

      {/* Dimensionality Reduction */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Dimensionality Reduction
          </Typography>

          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            {/* Use PCA Toggle */}
            <Box>
              <FormControlLabel
                control={
                  <Switch
                    checked={parameters.use_pca}
                    onChange={(e) => handleParameterChange('use_pca', e.target.checked)}
                    aria-label="Toggle PCA preprocessing"
                    tabIndex={0}
                  />
                }
                label="Apply PCA preprocessing"
              />
              <Typography variant="body2" color="text.secondary">
                Reduce dimensionality before clustering (recommended for high-dimensional data)
              </Typography>
            </Box>

            {/* Dimension Reducer Selector */}
            <FormControl fullWidth>
              <InputLabel>Visualization Method</InputLabel>
              <Select
                value={parameters.dim_reducer}
                onChange={(e) => handleParameterChange('dim_reducer', e.target.value)}
                aria-label="Select dimensionality reduction method for visualization"
                tabIndex={0}
              >
                {DIM_REDUCERS.map((reducer) => (
                  <MenuItem key={reducer.id} value={reducer.id}>
                    <Box>
                      <Typography variant="body1">{reducer.label}</Typography>
                      <Typography variant="caption" color="text.secondary">
                        {reducer.description}
                      </Typography>
                    </Box>
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Box>
        </CardContent>
      </Card>

      {/* Advanced Algorithm Parameters */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <Tune sx={{ mr: 1 }} />
            <Typography variant="h6">
              Algorithm Parameters
            </Typography>
          </Box>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Configure detailed parameters for each clustering algorithm
          </Typography>
          
          {/* <ParameterEditor onParametersChange={handleAdvancedParametersChange} /> */}
          <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic' }}>
            Advanced parameter configuration coming soon
          </Typography>
        </CardContent>
      </Card>

      {/* Validation and Run */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Ready to Run
          </Typography>

          {/* Validation Messages */}
          {!validation.isValid && (
            <Alert severity="error" sx={{ mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Please fix the following issues:
              </Typography>
              <ul style={{ margin: 0, paddingLeft: 20 }}>
                {validation.errors.map((error, index) => (
                  <li key={index}>
                    <Typography variant="body2">{error}</Typography>
                  </li>
                ))}
              </ul>
            </Alert>
          )}

          {/* Summary */}
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Configuration Summary:
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 1 }}>
              {parameters.methods.map((method) => (
                <Chip key={method} label={method} size="small" color="primary" />
              ))}
            </Box>
            <Typography variant="body2" color="text.secondary">
              {parameters.n_clusters} clusters • {parameters.sigma} sigma • {parameters.n_neighbors} neighbors
              {parameters.use_pca && ' • PCA enabled'}
            </Typography>
          </Box>

          <Button
            variant="contained"
            size="large"
            startIcon={<PlayArrow />}
            onClick={handleRunClustering}
            disabled={!canRunClustering()}
            aria-label="Start clustering analysis with selected parameters"
            tabIndex={0}
            fullWidth
          >
            Run Clustering Analysis
          </Button>
        </CardContent>
      </Card>
    </Box>
  );
};

export default ConfigPanel;
