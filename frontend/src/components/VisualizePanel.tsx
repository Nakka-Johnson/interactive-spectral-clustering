import React, { useState, useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  ToggleButton,
  ToggleButtonGroup,
  FormControl,
  Select,
  MenuItem,
  InputLabel,
  Chip,
  IconButton,
  Tooltip,
  Alert,
  Slider,
  Switch,
  FormControlLabel,
} from '@mui/material';
import {
  ScatterPlot,
  ThreeDRotation,
  Download,
  Fullscreen,
  Settings,
} from '@mui/icons-material';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { useClusteringStore } from '../store';

// Enhanced color palettes for clusters
const COLOR_PALETTES = {
  default: [
    '#5B8CFF', '#9C6EFF', '#4BD37B', '#FFB020', '#FF5470',
    '#00D4AA', '#FF8A00', '#8B5CF6', '#06B6D4', '#F59E0B',
    '#EF4444', '#10B981', '#3B82F6', '#8B5A2B', '#EC4899'
  ],
  viridis: [
    '#440154', '#482777', '#3F4A8A', '#31678E', '#26838F',
    '#1F9D8A', '#6CCE5A', '#B6DE2B', '#FEE825', '#440154',
    '#482777', '#3F4A8A', '#31678E', '#26838F', '#1F9D8A'
  ],
  plasma: [
    '#0C0786', '#5302A3', '#8B0AA5', '#B83289', '#DB5C68',
    '#F48849', '#FEBD2A', '#F0F921', '#0C0786', '#5302A3',
    '#8B0AA5', '#B83289', '#DB5C68', '#F48849', '#FEBD2A'
  ],
  rainbow: [
    '#FF0000', '#FF8000', '#FFFF00', '#80FF00', '#00FF00',
    '#00FF80', '#00FFFF', '#0080FF', '#0000FF', '#8000FF',
    '#FF00FF', '#FF0080', '#FF4040', '#40FF40', '#4040FF'
  ]
};

// Legacy cluster colors for backward compatibility
const CLUSTER_COLORS = COLOR_PALETTES.default;

interface VisualizationConfig {
  plotType: '2d' | '3d';
  algorithm: string;
  showCentroids: boolean;
  showConvexHulls: boolean;
  showGrid: boolean;
  pointSize: number;
  opacity: number;
  colorScheme: 'default' | 'viridis' | 'plasma' | 'rainbow';
}

const VisualizePanel: React.FC = () => {
  const { currentResult, parameters } = useClusteringStore();
  
  const [config, setConfig] = useState<VisualizationConfig>({
    plotType: '2d',
    algorithm: parameters.methods[0] || 'kmeans',
    showCentroids: true,
    showConvexHulls: false,
    showGrid: true,
    pointSize: 6,
    opacity: 0.8,
    colorScheme: 'viridis',
  });

  // Mock data generation for visualization
  const mockData = useMemo(() => {
    if (!currentResult) {
      // Generate mock data for preview
      const data = [];
      const numPoints = 200;
      const numClusters = parameters.n_clusters;
      
      for (let i = 0; i < numPoints; i++) {
        const cluster = Math.floor(Math.random() * numClusters);
        const angle = Math.random() * 2 * Math.PI;
        const radius = Math.random() * 2 + cluster * 3;
        
        data.push({
          x: Math.cos(angle) * radius + cluster * 5,
          y: Math.sin(angle) * radius + cluster * 5,
          z: Math.random() * 10,
          cluster: cluster,
          original_index: i,
        });
      }
      return data;
    }
    
    // In real app, this would process currentResult.embedding and currentResult.labels
    return [];
  }, [currentResult, parameters.n_clusters]);

  const availableAlgorithms = useMemo(() => {
    if (currentResult?.labels) {
      return Object.keys(currentResult.labels);
    }
    return parameters.methods;
  }, [currentResult, parameters.methods]);

  const handleConfigChange = (key: keyof VisualizationConfig, value: any) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  };

  const exportVisualization = (format: 'png' | 'svg' | 'pdf') => {
    // In real app, this would export the current visualization
    console.log(`Exporting visualization as ${format}`);
  };

  const enterFullscreen = () => {
    // In real app, this would enter fullscreen mode
    console.log('Entering fullscreen mode');
  };

  if (!currentResult && parameters.methods.length === 0) {
    return (
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="h6" color="text.secondary">
          Run clustering analysis to see visualizations
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Cluster Visualization
      </Typography>
      
      {/* Controls */}
      <Card sx={{ 
        mb: 3,
        transition: 'all 0.2s ease-in-out',
        '&:hover': {
          transform: 'translateY(-2px)',
          boxShadow: (theme) => `0 4px 20px ${theme.palette.primary.main}20`,
        }
      }}>
        <CardContent>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, alignItems: 'center' }}>
            {/* Plot Type Toggle */}
            <ToggleButtonGroup
              value={config.plotType}
              exclusive
              onChange={(_, value) => value && handleConfigChange('plotType', value)}
              size="small"
              aria-label="Select plot type"
            >
              <ToggleButton value="2d" aria-label="Switch to 2D plot view" tabIndex={0}>
                <ScatterPlot sx={{ mr: 1 }} />
                2D Plot
              </ToggleButton>
              <ToggleButton value="3d" aria-label="Switch to 3D plot view" tabIndex={0}>
                <ThreeDRotation sx={{ mr: 1 }} />
                3D Plot
              </ToggleButton>
            </ToggleButtonGroup>

            {/* Algorithm Selection */}
            <FormControl size="small" sx={{ minWidth: 150 }}>
              <InputLabel>Algorithm</InputLabel>
              <Select
                value={config.algorithm}
                onChange={(e) => handleConfigChange('algorithm', e.target.value)}
                aria-label="Select clustering algorithm to visualize"
                tabIndex={0}
              >
                {availableAlgorithms.map((alg) => (
                  <MenuItem key={alg} value={alg}>
                    {alg.toUpperCase()}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {/* Display Options */}
            <FormControlLabel
              control={
                <Switch
                  checked={config.showCentroids}
                  onChange={(e) => handleConfigChange('showCentroids', e.target.checked)}
                  size="small"
                  aria-label="Toggle cluster centroids visibility"
                  tabIndex={0}
                />
              }
              label="Centroids"
            />

            <FormControlLabel
              control={
                <Switch
                  checked={config.showConvexHulls}
                  onChange={(e) => handleConfigChange('showConvexHulls', e.target.checked)}
                  size="small"
                  aria-label="Toggle convex hulls visibility"
                  tabIndex={0}
                />
              }
              label="Hulls"
            />

            <FormControlLabel
              control={
                <Switch
                  checked={config.showGrid}
                  onChange={(e) => handleConfigChange('showGrid', e.target.checked)}
                  size="small"
                  aria-label="Toggle grid visibility"
                  tabIndex={0}
                />
              }
              label="Grid"
            />

            {/* Export Options */}
            <Box sx={{ ml: 'auto', display: 'flex', gap: 1 }}>
              <Tooltip title="Export as PNG">
                <IconButton 
                  onClick={() => exportVisualization('png')} 
                  size="small"
                  aria-label="Export visualization as PNG image"
                  tabIndex={0}
                >
                  <Download />
                </IconButton>
              </Tooltip>
              <Tooltip title="Fullscreen">
                <IconButton 
                  onClick={enterFullscreen} 
                  size="small"
                  aria-label="Enter fullscreen mode"
                  tabIndex={0}
                >
                  <Fullscreen />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>
        </CardContent>
      </Card>

      {/* Main Visualization */}
      <Card sx={{ 
        mb: 3,
        transition: 'all 0.2s ease-in-out',
        '&:hover': {
          transform: 'translateY(-2px)',
          boxShadow: (theme) => `0 4px 20px ${theme.palette.primary.main}20`,
        }
      }}>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">
              {config.algorithm.toUpperCase()} - {config.plotType.toUpperCase()} Visualization
            </Typography>
            <Box sx={{ display: 'flex', gap: 1 }}>
              {mockData.length > 0 && (
                <Chip
                  label={`${mockData.length} points, ${parameters.n_clusters} clusters`}
                  size="small"
                  color="primary"
                />
              )}
            </Box>
          </Box>

          {config.plotType === '2d' ? (
            <Box sx={{ height: 500, width: '100%' }}>
              <ResponsiveContainer>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" stroke={config.showGrid ? '#ccc' : 'transparent'} />
                  <XAxis 
                    type="number" 
                    dataKey="x"
                    domain={['dataMin - 1', 'dataMax + 1']}
                    label={{ value: 'Component 1', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis 
                    type="number" 
                    dataKey="y"
                    domain={['dataMin - 1', 'dataMax + 1']}
                    label={{ value: 'Component 2', angle: -90, position: 'insideLeft' }}
                  />
                  <RechartsTooltip
                    formatter={(value: any, name: string) => [
                      typeof value === 'number' ? value.toFixed(3) : value,
                      name
                    ]}
                  />
                  <Legend />
                  
                  {/* Render points by cluster */}
                  {Array.from(new Set(mockData.map(d => d.cluster))).map((cluster, index) => {
                    const selectedPalette = COLOR_PALETTES[config.colorScheme];
                    return (
                      <Scatter
                        key={cluster}
                        name={`Cluster ${cluster}`}
                        data={mockData.filter(d => d.cluster === cluster)}
                        fill={selectedPalette[index % selectedPalette.length]}
                        fillOpacity={config.opacity}
                        strokeWidth={1}
                        stroke="#fff"
                      />
                    );
                  })}
                </ScatterChart>
              </ResponsiveContainer>
            </Box>
          ) : (
            <Box 
              sx={{ 
                height: 500, 
                bgcolor: 'grey.50', 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center',
                borderRadius: 1,
                border: '1px dashed',
                borderColor: 'grey.300'
              }}
            >
              <Box sx={{ textAlign: 'center' }}>
                <ThreeDRotation sx={{ fontSize: 64, color: 'grey.400', mb: 1 }} />
                <Typography variant="h6" color="text.secondary">
                  3D Visualization
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Interactive 3D plot would be rendered here using Three.js/React Three Fiber
                </Typography>
              </Box>
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Visualization Settings */}
      <Card sx={{
        transition: 'all 0.2s ease-in-out',
        '&:hover': {
          transform: 'translateY(-2px)',
          boxShadow: (theme) => `0 4px 20px ${theme.palette.primary.main}20`,
        }
      }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            <Settings sx={{ mr: 1, verticalAlign: 'bottom' }} />
            Visualization Settings
          </Typography>
          
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            {/* Point Size */}
            <Box>
              <Typography gutterBottom>Point Size</Typography>
              <Slider
                value={config.pointSize}
                onChange={(_, value) => handleConfigChange('pointSize', value)}
                min={2}
                max={15}
                step={1}
                marks
                valueLabelDisplay="auto"
              />
            </Box>

            {/* Opacity */}
            <Box>
              <Typography gutterBottom>Opacity</Typography>
              <Slider
                value={config.opacity}
                onChange={(_, value) => handleConfigChange('opacity', value)}
                min={0.1}
                max={1}
                step={0.1}
                marks
                valueLabelDisplay="auto"
              />
            </Box>

            {/* Color Scheme */}
            <FormControl>
              <InputLabel>Color Scheme</InputLabel>
              <Select
                value={config.colorScheme}
                onChange={(e) => handleConfigChange('colorScheme', e.target.value)}
                sx={{
                  '& .MuiOutlinedInput-notchedOutline': {
                    transition: 'border-color 0.2s ease-in-out',
                  },
                  '&:hover .MuiOutlinedInput-notchedOutline': {
                    borderColor: 'primary.main',
                  },
                  '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                    borderColor: 'primary.main',
                    borderWidth: '2px',
                  },
                }}
              >
                <MenuItem 
                  value="default"
                  sx={{
                    '&:hover': {
                      bgcolor: (theme) => theme.palette.action.hover,
                      transform: 'translateX(2px)',
                      transition: 'all 0.2s ease-in-out',
                    },
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Box sx={{ display: 'flex', gap: 0.5 }}>
                      {COLOR_PALETTES.default.slice(0, 5).map((color, i) => (
                        <Box
                          key={i}
                          sx={{
                            width: 12,
                            height: 12,
                            bgcolor: color,
                            borderRadius: 0.5,
                            border: '1px solid rgba(255,255,255,0.1)',
                          }}
                        />
                      ))}
                    </Box>
                    Default
                  </Box>
                </MenuItem>
                <MenuItem 
                  value="viridis"
                  sx={{
                    '&:hover': {
                      bgcolor: (theme) => theme.palette.action.hover,
                      transform: 'translateX(2px)',
                      transition: 'all 0.2s ease-in-out',
                    },
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Box sx={{ display: 'flex', gap: 0.5 }}>
                      {COLOR_PALETTES.viridis.slice(0, 5).map((color, i) => (
                        <Box
                          key={i}
                          sx={{
                            width: 12,
                            height: 12,
                            bgcolor: color,
                            borderRadius: 0.5,
                            border: '1px solid rgba(255,255,255,0.1)',
                          }}
                        />
                      ))}
                    </Box>
                    Viridis
                  </Box>
                </MenuItem>
                <MenuItem 
                  value="plasma"
                  sx={{
                    '&:hover': {
                      bgcolor: (theme) => theme.palette.action.hover,
                      transform: 'translateX(2px)',
                      transition: 'all 0.2s ease-in-out',
                    },
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Box sx={{ display: 'flex', gap: 0.5 }}>
                      {COLOR_PALETTES.plasma.slice(0, 5).map((color, i) => (
                        <Box
                          key={i}
                          sx={{
                            width: 12,
                            height: 12,
                            bgcolor: color,
                            borderRadius: 0.5,
                            border: '1px solid rgba(255,255,255,0.1)',
                          }}
                        />
                      ))}
                    </Box>
                    Plasma
                  </Box>
                </MenuItem>
                <MenuItem 
                  value="rainbow"
                  sx={{
                    '&:hover': {
                      bgcolor: (theme) => theme.palette.action.hover,
                      transform: 'translateX(2px)',
                      transition: 'all 0.2s ease-in-out',
                    },
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Box sx={{ display: 'flex', gap: 0.5 }}>
                      {COLOR_PALETTES.rainbow.slice(0, 5).map((color, i) => (
                        <Box
                          key={i}
                          sx={{
                            width: 12,
                            height: 12,
                            bgcolor: color,
                            borderRadius: 0.5,
                            border: '1px solid rgba(255,255,255,0.1)',
                          }}
                        />
                      ))}
                    </Box>
                    Rainbow
                  </Box>
                </MenuItem>
              </Select>
            </FormControl>
          </Box>
        </CardContent>
      </Card>

      {!currentResult && (
        <Alert severity="info" sx={{ mt: 2 }}>
          This is a preview with mock data. Run clustering analysis to see actual results.
        </Alert>
      )}
    </Box>
  );
};

export default VisualizePanel;
