/**
 * ConfigPage component for Interactive Spectral Clustering Platform.
 * Provides comprehensive configuration for clustering algorithms and preprocessing.
 */

import React, { useState } from 'react';
import {
  Box,
  Container,
  Typography,
  Card,
  CardContent,
  Switch,
  FormControlLabel,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  Stack,
  Chip,
  Divider,
  Alert,
  Paper,
  Toolbar,
} from '@mui/material';
import Grid from '@mui/material/Grid';
import {
  TuneOutlined,
  PlayArrow,
  ClearAll,
  Psychology,
  Transform,
} from '@mui/icons-material';
import AlgorithmConfigPanel from '../../components/AlgorithmConfigPanel';
import { runClustering } from '../../api';

interface DimensionalityReductionConfig {
  enablePCA: boolean;
  components: number;
  visualizationMethod: 'PCA' | 't-SNE' | 'UMAP';
}

interface PreprocessingConfig {
  normalize: boolean;
  standardize: boolean;
  removeOutliers: boolean;
}

export const ConfigPage: React.FC = () => {
  const [selectedAlgorithms, setSelectedAlgorithms] = useState<any[]>([]);
  const [dimensionalityConfig, setDimensionalityConfig] = useState<DimensionalityReductionConfig>({
    enablePCA: true,
    components: 10,
    visualizationMethod: 'PCA',
  });
  const [preprocessingConfig, setPreprocessingConfig] = useState<PreprocessingConfig>({
    normalize: true,
    standardize: false,
    removeOutliers: false,
  });

  const handleAlgorithmConfigChange = (configs: any[]) => {
    setSelectedAlgorithms(configs.filter((config) => config.selected));
  };

  const handleDimensionalityChange = (field: keyof DimensionalityReductionConfig, value: any) => {
    setDimensionalityConfig((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const handlePreprocessingChange = (field: keyof PreprocessingConfig, value: boolean) => {
    setPreprocessingConfig((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleClearResults = () => {
    // Clear previous results
    console.log('Clearing previous results...');
  };

  const handleRunClustering = async () => {
    // Validate that at least one algorithm is selected
    if (selectedAlgorithms.length === 0) {
      alert('Please select at least one clustering algorithm.');
      return;
    }

    // Prepare configuration for API call
    const clusteringConfig = {
      algorithms: selectedAlgorithms.map(alg => ({
        name: alg.id,
        parameters: alg.parameters
      })),
      dimensionality_reduction: dimensionalityConfig,
      preprocessing: preprocessingConfig,
    };

    try {
      // Call the clustering API
      const result = await runClustering({
        dataset_id: 'current', // This should come from app state
        algorithm: selectedAlgorithms[0].id, // Primary algorithm
        ...selectedAlgorithms[0].parameters,
        preprocessing: preprocessingConfig,
        dimensionality_reduction: dimensionalityConfig
      });
      
      alert(`Clustering analysis completed! Results available for visualization.`);
    } catch (error) {
      console.error('Failed to start clustering analysis:', error);
      alert('Failed to start clustering analysis. Please try again.');
    }
  };

  return (
    <Box sx={{ minHeight: '100vh', backgroundColor: 'background.default' }}>
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Stack spacing={4}>
          {/* Page Header */}
          <Box>
            <Typography variant="h4" gutterBottom>
              Clustering Configuration
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Configure algorithms, parameters, and preprocessing options for your clustering
              analysis.
            </Typography>
          </Box>

          {/* Algorithm Configuration */}
          <AlgorithmConfigPanel onConfigChange={handleAlgorithmConfigChange} />

          {/* Dimensionality Reduction */}
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" gap={2} mb={3}>
                <Psychology color="primary" fontSize="large" />
                <Typography variant="h6">Dimensionality Reduction</Typography>
              </Box>

              <Grid container spacing={3}>
                <Grid item xs={12} md={4}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={dimensionalityConfig.enablePCA}
                        onChange={(e) => handleDimensionalityChange('enablePCA', e.target.checked)}
                      />
                    }
                    label="Enable PCA"
                  />
                  <Typography variant="caption" display="block" color="text.secondary">
                    Apply Principal Component Analysis for dimensionality reduction
                  </Typography>
                </Grid>

                <Grid item xs={12} md={4}>
                  <Typography gutterBottom>
                    Components: {dimensionalityConfig.components}
                  </Typography>
                  <Slider
                    value={dimensionalityConfig.components}
                    onChange={(_, value) => handleDimensionalityChange('components', value)}
                    min={2}
                    max={50}
                    marks={[
                      { value: 2, label: '2' },
                      { value: 10, label: '10' },
                      { value: 25, label: '25' },
                      { value: 50, label: '50' },
                    ]}
                    disabled={!dimensionalityConfig.enablePCA}
                  />
                  <Typography variant="caption" color="text.secondary">
                    Number of principal components to retain
                  </Typography>
                </Grid>

                <Grid item xs={12} md={4}>
                  <FormControl fullWidth>
                    <InputLabel>Visualization Method</InputLabel>
                    <Select
                      value={dimensionalityConfig.visualizationMethod}
                      label="Visualization Method"
                      onChange={(e) =>
                        handleDimensionalityChange('visualizationMethod', e.target.value)
                      }
                    >
                      <MenuItem value="PCA">PCA</MenuItem>
                      <MenuItem value="t-SNE">t-SNE</MenuItem>
                      <MenuItem value="UMAP">UMAP</MenuItem>
                    </Select>
                  </FormControl>
                  <Typography variant="caption" display="block" color="text.secondary">
                    Method for 2D/3D visualization
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>

          {/* Data Preprocessing */}
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" gap={2} mb={3}>
                <Transform color="primary" fontSize="large" />
                <Typography variant="h6">Data Preprocessing</Typography>
              </Box>

              <Grid container spacing={3}>
                <Grid item xs={12} md={4}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={preprocessingConfig.normalize}
                        onChange={(e) => handlePreprocessingChange('normalize', e.target.checked)}
                      />
                    }
                    label="Normalize"
                  />
                  <Typography variant="caption" display="block" color="text.secondary">
                    Scale features to [0, 1] range
                  </Typography>
                </Grid>

                <Grid item xs={12} md={4}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={preprocessingConfig.standardize}
                        onChange={(e) => handlePreprocessingChange('standardize', e.target.checked)}
                      />
                    }
                    label="Standardize"
                  />
                  <Typography variant="caption" display="block" color="text.secondary">
                    Center data to mean=0, std=1
                  </Typography>
                </Grid>

                <Grid item xs={12} md={4}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={preprocessingConfig.removeOutliers}
                        onChange={(e) =>
                          handlePreprocessingChange('removeOutliers', e.target.checked)
                        }
                      />
                    }
                    label="Remove Outliers"
                  />
                  <Typography variant="caption" display="block" color="text.secondary">
                    Filter outliers using IQR method
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>

          {/* Configuration Summary */}
          {selectedAlgorithms.length > 0 && (
            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Configuration Summary
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle2" gutterBottom>
                      Selected Algorithms ({selectedAlgorithms.length})
                    </Typography>
                    <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                      {selectedAlgorithms.map((alg) => (
                        <Chip
                          key={alg.id}
                          label={alg.name}
                          color="primary"
                          variant="outlined"
                          size="small"
                        />
                      ))}
                    </Stack>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle2" gutterBottom>
                      Preprocessing Steps
                    </Typography>
                    <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                      {preprocessingConfig.normalize && (
                        <Chip label="Normalize" size="small" color="secondary" variant="outlined" />
                      )}
                      {preprocessingConfig.standardize && (
                        <Chip
                          label="Standardize"
                          size="small"
                          color="secondary"
                          variant="outlined"
                        />
                      )}
                      {preprocessingConfig.removeOutliers && (
                        <Chip
                          label="Remove Outliers"
                          size="small"
                          color="secondary"
                          variant="outlined"
                        />
                      )}
                      {dimensionalityConfig.enablePCA && (
                        <Chip
                          label={`PCA (${dimensionalityConfig.components} components)`}
                          size="small"
                          color="info"
                          variant="outlined"
                        />
                      )}
                    </Stack>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          )}

          {/* Validation */}
          {selectedAlgorithms.length === 0 && (
            <Alert severity="info">
              Please select at least one clustering algorithm to proceed with the analysis.
            </Alert>
          )}
        </Stack>
      </Container>

      {/* Bottom Action Bar */}
      <Paper
        elevation={8}
        sx={{
          position: 'fixed',
          bottom: 0,
          left: 0,
          right: 0,
          zIndex: 1000,
          borderRadius: 0,
        }}
      >
        <Toolbar>
          <Container maxWidth="lg">
            <Box display="flex" justifyContent="space-between" alignItems="center" width="100%">
              <Button
                variant="outlined"
                startIcon={<ClearAll />}
                onClick={handleClearResults}
                sx={{ color: 'text.secondary' }}
              >
                Clear Previous Results
              </Button>

              <Button
                variant="contained"
                size="large"
                startIcon={<PlayArrow />}
                onClick={handleRunClustering}
                disabled={selectedAlgorithms.length === 0}
              >
                Run Clustering Analysis
              </Button>
            </Box>
          </Container>
        </Toolbar>
      </Paper>

      {/* Add bottom padding to prevent content overlap with fixed toolbar */}
      <Box sx={{ height: 80 }} />
    </Box>
  );
};

export default ConfigPage;
