/**
 * VisualizePage component for Interactive Spectral Clustering Platform.
 *
 * Comprehensive visualization page with multiple tabs, algorithm comparison,
 * and interactive controls for clustering results analysis.
 */

import React, { useState, useEffect, useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Alert,
  Button,
  Stack,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Slider,
  Tabs,
  Tab,
  Grid,
  Chip,
  Paper,
  Divider,
  LinearProgress,
} from '@mui/material';
import {
  Refresh,
  Download,
  Palette,
  Visibility,
  Analytics,
  Assessment,
  ScatterPlot,
  ThreeDRotation,
  BarChart,
  TrendingUp,
} from '@mui/icons-material';
import { BarChart as RechartsBarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';
import Cluster2D from './Cluster2D_Enhanced';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`visualization-tabpanel-${index}`}
      aria-labelledby={`visualization-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

function a11yProps(index: number) {
  return {
    id: `visualization-tab-${index}`,
    'aria-controls': `visualization-tabpanel-${index}`,
  };
}

// Mock data for demonstration - in real app this would come from API
const mockClusteringResults = {
  kmeans: {
    labels: [0, 0, 1, 1, 2, 2, 0, 1, 2, 0],
    embedding: [[1, 2], [1.5, 2.1], [5, 6], [5.2, 5.8], [9, 10], [9.1, 9.8], [1.2, 2.3], [5.1, 6.2], [8.9, 9.7], [1.1, 2.0]],
    metrics: { silhouette: 0.67, calinski_harabasz: 156.3, davies_bouldin: 0.89, execution_time: 0.045 },
    centroids: [[1.2, 2.1], [5.1, 6.0], [9.0, 9.8]]
  },
  spectral: {
    labels: [0, 0, 1, 1, 2, 2, 0, 1, 2, 0],
    embedding: [[1.1, 2.2], [1.4, 2.0], [5.1, 6.1], [5.3, 5.9], [8.8, 9.9], [9.2, 9.9], [1.3, 2.4], [5.0, 6.1], [8.7, 9.6], [1.0, 1.9]],
    metrics: { silhouette: 0.73, calinski_harabasz: 189.7, davies_bouldin: 0.76, execution_time: 0.123 },
    centroids: [[1.2, 2.1], [5.1, 6.0], [8.9, 9.8]]
  },
  dbscan: {
    labels: [0, 0, 1, 1, -1, 2, 0, 1, 2, 0],
    embedding: [[1.1, 2.1], [1.6, 2.2], [5.2, 6.0], [5.1, 5.7], [7.5, 8.0], [9.0, 9.9], [1.1, 2.2], [5.3, 6.3], [9.1, 9.8], [1.2, 2.1]],
    metrics: { silhouette: 0.58, calinski_harabasz: 134.2, davies_bouldin: 1.12, execution_time: 0.087 },
    centroids: null
  }
};

const colorPalettes = {
  viridis: ['#440154', '#31688e', '#35b779', '#fde725'],
  plasma: ['#0d0887', '#6a00a8', '#b12a90', '#e16462', '#fca636'],
  cool: ['#6baed6', '#4292c6', '#2171b5', '#084594'],
  warm: ['#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#b10026']
};

/**
 * VisualizePage component.
 */
const VisualizePage: React.FC = () => {
  // State management
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string>('kmeans');
  const [selectedPalette, setSelectedPalette] = useState<string>('viridis');
  const [showLabels, setShowLabels] = useState(true);
  const [pointSize, setPointSize] = useState(6);
  const [pointOpacity, setPointOpacity] = useState(0.8);
  const [activeTab, setActiveTab] = useState(0);
  const [animationSpeed, setAnimationSpeed] = useState(1.0);

  // Get current algorithm results
  const currentResults = useMemo(() => {
    return mockClusteringResults[selectedAlgorithm as keyof typeof mockClusteringResults];
  }, [selectedAlgorithm]);

  // Generate cluster statistics
  const clusterStats = useMemo(() => {
    if (!currentResults) return [];
    
    const labelCounts = currentResults.labels.reduce((acc, label) => {
      acc[label] = (acc[label] || 0) + 1;
      return acc;
    }, {} as Record<number, number>);

    return Object.entries(labelCounts).map(([label, count]) => ({
      cluster: parseInt(label),
      count,
      percentage: (count / currentResults.labels.length * 100).toFixed(1)
    }));
  }, [currentResults]);

  // Performance comparison data
  const performanceData = useMemo(() => {
    return Object.entries(mockClusteringResults).map(([algorithm, results]) => ({
      algorithm: algorithm.toUpperCase(),
      silhouette: results.metrics.silhouette,
      calinski_harabasz: results.metrics.calinski_harabasz / 100, // Scale for visualization
      davies_bouldin: 2 - results.metrics.davies_bouldin, // Invert (lower is better)
      execution_time: results.metrics.execution_time * 1000 // Convert to ms
    }));
  }, []);

  // Elbow curve data (mock)
  const elbowData = useMemo(() => {
    return [
      { k: 1, wcss: 250 },
      { k: 2, wcss: 120 },
      { k: 3, wcss: 78 },
      { k: 4, wcss: 65 },
      { k: 5, wcss: 58 },
      { k: 6, wcss: 55 },
      { k: 7, wcss: 54 },
      { k: 8, wcss: 53 }
    ];
  }, []);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handleRefresh = () => {
    // Placeholder for refresh functionality
    console.log('Refreshing visualizations...');
  };

  const handleExportVisualization = () => {
    // Placeholder for export visualization
    console.log('Exporting visualization...');
    alert('Export visualization functionality would be implemented here');
  };

  const handleExportMetrics = () => {
    // Placeholder for export metrics
    console.log('Exporting metrics...');
    alert('Export metrics functionality would be implemented here');
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          Clustering Visualization
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Interactive visualization and analysis of clustering results across different algorithms.
        </Typography>
      </Box>

      {/* Visualization Settings Card */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box display="flex" alignItems="center" gap={2} mb={3}>
            <Palette color="primary" fontSize="large" />
            <Typography variant="h6">Visualization Settings</Typography>
          </Box>

          <Grid container spacing={3}>
            <Grid item xs={12} md={3}>
              <FormControl fullWidth>
                <InputLabel>Algorithm Results</InputLabel>
                <Select
                  value={selectedAlgorithm}
                  label="Algorithm Results"
                  onChange={(e) => setSelectedAlgorithm(e.target.value)}
                >
                  <MenuItem value="kmeans">K-Means</MenuItem>
                  <MenuItem value="spectral">Spectral Clustering</MenuItem>
                  <MenuItem value="dbscan">DBSCAN</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} md={3}>
              <FormControl fullWidth>
                <InputLabel>Color Palette</InputLabel>
                <Select
                  value={selectedPalette}
                  label="Color Palette"
                  onChange={(e) => setSelectedPalette(e.target.value)}
                >
                  <MenuItem value="viridis">Viridis (Default)</MenuItem>
                  <MenuItem value="plasma">Plasma</MenuItem>
                  <MenuItem value="cool">Cool</MenuItem>
                  <MenuItem value="warm">Warm</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} md={2}>
              <FormControlLabel
                control={
                  <Switch
                    checked={showLabels}
                    onChange={(e) => setShowLabels(e.target.checked)}
                  />
                }
                label="Show Labels"
              />
            </Grid>

            <Grid item xs={12} md={4}>
              <Typography gutterBottom>Point Size: {pointSize}</Typography>
              <Slider
                value={pointSize}
                onChange={(_, value) => setPointSize(value as number)}
                min={2}
                max={12}
                marks={[
                  { value: 2, label: '2' },
                  { value: 6, label: '6' },
                  { value: 12, label: '12' }
                ]}
              />
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Tabs */}
      <Card>
        <CardContent>
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs value={activeTab} onChange={handleTabChange} aria-label="visualization tabs">
              <Tab label="2D Scatter Plot" icon={<ScatterPlot />} {...a11yProps(0)} />
              <Tab label="3D Visualization" icon={<ThreeDRotation />} {...a11yProps(1)} />
              <Tab label="Cluster Analysis" icon={<Analytics />} {...a11yProps(2)} />
              <Tab label="Advanced Metrics" icon={<Assessment />} {...a11yProps(3)} />
            </Tabs>
          </Box>

          {/* 2D Scatter Plot Tab */}
          <TabPanel value={activeTab} index={0}>
            <Grid container spacing={3}>
              <Grid item xs={12} lg={8}>
                <Cluster2D
                  points={currentResults?.embedding}
                  labels={currentResults?.labels}
                  colors={colorPalettes[selectedPalette as keyof typeof colorPalettes]}
                  title={`${selectedAlgorithm.toUpperCase()} Clustering Results`}
                  pointSize={pointSize}
                  showLegend={showLabels}
                />
              </Grid>
              
              <Grid item xs={12} lg={4}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Legend
                    </Typography>
                    <Stack spacing={2}>
                      {clusterStats.map((stat) => (
                        <Box key={stat.cluster} display="flex" alignItems="center" gap={2}>
                          <Box
                            sx={{
                              width: 16,
                              height: 16,
                              borderRadius: '50%',
                              backgroundColor: colorPalettes[selectedPalette as keyof typeof colorPalettes][stat.cluster] || '#666'
                            }}
                          />
                          <Typography variant="body2">
                            {stat.cluster === -1 ? 'Cluster undefined' : `Cluster ${stat.cluster}`}: {stat.count} points ({stat.percentage}%)
                          </Typography>
                        </Box>
                      ))}
                      
                      {currentResults?.centroids && (
                        <Box display="flex" alignItems="center" gap={2}>
                          <Box
                            sx={{
                              width: 16,
                              height: 16,
                              backgroundColor: 'red',
                              borderRadius: 1
                            }}
                          />
                          <Typography variant="body2">
                            Centroids
                          </Typography>
                        </Box>
                      )}
                    </Stack>

                    <Divider sx={{ my: 2 }} />
                    
                    <Typography variant="subtitle2" gutterBottom>
                      Controls
                    </Typography>
                    <Typography gutterBottom>Opacity: {Math.round(pointOpacity * 100)}%</Typography>
                    <Slider
                      value={pointOpacity}
                      onChange={(_, value) => setPointOpacity(value as number)}
                      min={0.1}
                      max={1.0}
                      step={0.1}
                      size="small"
                    />
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>

          {/* 3D Visualization Tab */}
          <TabPanel value={activeTab} index={1}>
            <Grid container spacing={3}>
              <Grid item xs={12} lg={8}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      3D Interactive Scatter Plot
                    </Typography>
                    <Box
                      sx={{
                        height: 400,
                        bgcolor: 'background.default',
                        borderRadius: 1,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        border: 1,
                        borderColor: 'divider'
                      }}
                    >
                      <Typography color="text.secondary">
                        3D visualization with @react-three/fiber would be implemented here
                      </Typography>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} lg={4}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      3D Controls
                    </Typography>
                    
                    <Stack spacing={2}>
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          Stats: {currentResults?.labels.length} points
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Color scheme: {selectedPalette}
                        </Typography>
                      </Box>
                      
                      <Box>
                        <Typography gutterBottom>Animation Speed: {animationSpeed.toFixed(1)}x</Typography>
                        <Slider
                          value={animationSpeed}
                          onChange={(_, value) => setAnimationSpeed(value as number)}
                          min={0.1}
                          max={3.0}
                          step={0.1}
                          marks={[
                            { value: 0.1, label: '0.1x' },
                            { value: 1.0, label: '1x' },
                            { value: 3.0, label: '3x' }
                          ]}
                        />
                      </Box>
                      
                      <Alert severity="info">
                        Mouse controls: Left click to rotate, wheel to zoom, right click to pan
                      </Alert>
                    </Stack>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>

          {/* Cluster Analysis Tab */}
          <TabPanel value={activeTab} index={2}>
            {/* Cluster Summary Cards */}
            <Grid container spacing={2} sx={{ mb: 3 }}>
              {clusterStats.map((stat) => (
                <Grid item xs={12} sm={6} md={3} key={stat.cluster}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        {stat.cluster === -1 ? 'Cluster undefined' : `Cluster ${stat.cluster}`}
                      </Typography>
                      <Typography variant="h4" color="primary">
                        {stat.count}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {stat.percentage}% of total points
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>

            {/* Algorithm Performance Summary */}
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Per-Algorithm Performance Summary
                </Typography>
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle2" gutterBottom>
                      Silhouette Score Comparison
                    </Typography>
                    <ResponsiveContainer width="100%" height={200}>
                      <RechartsBarChart data={performanceData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="algorithm" />
                        <YAxis domain={[0, 1]} />
                        <Tooltip />
                        <Bar dataKey="silhouette" fill="#8884d8" />
                      </RechartsBarChart>
                    </ResponsiveContainer>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle2" gutterBottom>
                      Execution Time (ms)
                    </Typography>
                    <ResponsiveContainer width="100%" height={200}>
                      <RechartsBarChart data={performanceData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="algorithm" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="execution_time" fill="#82ca9d" />
                      </RechartsBarChart>
                    </ResponsiveContainer>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </TabPanel>

          {/* Advanced Metrics Tab */}
          <TabPanel value={activeTab} index={3}>
            <Grid container spacing={3}>
              {/* Silhouette Analysis */}
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Silhouette Analysis
                    </Typography>
                    <Typography variant="subtitle2" gutterBottom>
                      Per-Algorithm Comparison
                    </Typography>
                    <ResponsiveContainer width="100%" height={250}>
                      <RechartsBarChart data={performanceData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="algorithm" />
                        <YAxis domain={[0, 1]} />
                        <Tooltip />
                        <Bar dataKey="silhouette" fill="#8884d8" />
                      </RechartsBarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </Grid>

              {/* Elbow Method */}
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Elbow Method (WCSS vs K)
                    </Typography>
                    <ResponsiveContainer width="100%" height={250}>
                      <LineChart data={elbowData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="k" />
                        <YAxis />
                        <Tooltip />
                        <Line type="monotone" dataKey="wcss" stroke="#8884d8" strokeWidth={2} />
                      </LineChart>
                    </ResponsiveContainer>
                    <Alert severity="success" sx={{ mt: 2 }}>
                      Optimal K recommendation: K=3 (elbow point detected)
                    </Alert>
                  </CardContent>
                </Card>
              </Grid>

              {/* Computational Performance */}
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Computational Performance
                    </Typography>
                    <ResponsiveContainer width="100%" height={200}>
                      <RechartsBarChart data={performanceData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="algorithm" />
                        <YAxis />
                        <Tooltip formatter={(value) => [`${value} ms`, 'Execution Time']} />
                        <Bar dataKey="execution_time" fill="#ffc658" />
                      </RechartsBarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </Grid>

              {/* Scalability Analysis */}
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Scalability Analysis
                    </Typography>
                    <Stack spacing={2}>
                      <Alert severity="info">
                        <strong>K-Means:</strong> O(n×k×i×d) - Best for large datasets with spherical clusters
                      </Alert>
                      <Alert severity="warning">
                        <strong>Spectral:</strong> O(n³) - Expensive for large datasets but handles complex shapes
                      </Alert>
                      <Alert severity="success">
                        <strong>DBSCAN:</strong> O(n log n) - Good performance, handles noise and varying densities
                      </Alert>
                    </Stack>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>
        </CardContent>
      </Card>

      {/* Footer Buttons */}
      <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, mt: 3 }}>
        <Button variant="outlined" startIcon={<Refresh />} onClick={handleRefresh}>
          Refresh Visualizations
        </Button>
        <Button variant="outlined" startIcon={<Download />} onClick={handleExportVisualization}>
          Export Visualization
        </Button>
        <Button variant="outlined" startIcon={<BarChart />} onClick={handleExportMetrics}>
          Export Metrics
        </Button>
      </Box>
    </Box>
  );
};

export default VisualizePage;
