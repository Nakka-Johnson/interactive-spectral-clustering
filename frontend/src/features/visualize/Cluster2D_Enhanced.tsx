/**
 * Cluster2D component for Interactive Spectral Clustering Platform.
 * 
 * Enhanced to support both traditional 2D points and embedding points from
 * dimensionality reduction techniques (PCA, t-SNE, UMAP).
 */

import React, { useMemo } from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Alert,
  Chip,
  Stack
} from '@mui/material';

/**
 * Interface for embedding points from the backend.
 */
interface EmbeddingPoint {
  x: number;
  y: number;
  original_index: number;
}

/**
 * Interface for 2D data points.
 */
interface DataPoint {
  x: number;
  y: number;
  cluster: number;
  index: number;
}

/**
 * Props for Cluster2D component.
 */
interface Cluster2DProps {
  /** 2D coordinates for each data point */
  points?: number[][];
  /** Embedding points from dimensionality reduction */
  embeddings?: EmbeddingPoint[];
  /** Cluster labels for each point */
  labels?: number[];
  /** Custom colors for clusters */
  colors?: string[];
  /** Title for the visualization */
  title?: string;
  /** X-axis label */
  xLabel?: string;
  /** Y-axis label */
  yLabel?: string;
  /** Width of the chart */
  width?: number;
  /** Height of the chart */
  height?: number;
  /** Show legend */
  showLegend?: boolean;
  /** Show grid */
  showGrid?: boolean;
  /** Point size */
  pointSize?: number;
  /** Callback when point is clicked */
  onPointClick?: (point: DataPoint) => void;
  /** Embedding method name for labeling */
  embeddingMethod?: string;
  /** Loading state */
  loading?: boolean;
}

/**
 * Default color palette for clusters.
 */
const DEFAULT_COLORS = [
  '#8884d8', '#82ca9d', '#ffc658', '#ff7c7c', '#8dd1e1',
  '#d084d0', '#ffb347', '#87d068', '#ff6b6b', '#4ecdc4',
  '#45b7d1', '#f39c12', '#9b59b6', '#e74c3c', '#1abc9c'
];

/**
 * Generate cluster statistics.
 */
const generateClusterStats = (labels: number[]) => {
  const clusters = new Set(labels);
  const stats = Array.from(clusters).map(clusterId => {
    const count = labels.filter(label => label === clusterId).length;
    const percentage = ((count / labels.length) * 100).toFixed(1);
    return { clusterId, count, percentage };
  });
  
  return stats.sort((a, b) => a.clusterId - b.clusterId);
};

/**
 * Custom tooltip component for scatter plot.
 */
const CustomTooltip: React.FC<any> = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <Box
        sx={{
          bgcolor: 'background.paper',
          border: 1,
          borderColor: 'divider',
          borderRadius: 1,
          p: 1,
          boxShadow: 2
        }}
      >
        <Typography variant="body2">
          Point {data.index}
        </Typography>
        <Typography variant="body2">
          X: {data.x.toFixed(3)}
        </Typography>
        <Typography variant="body2">
          Y: {data.y.toFixed(3)}
        </Typography>
        <Typography variant="body2">
          Cluster: {data.cluster}
        </Typography>
      </Box>
    );
  }
  return null;
};

/**
 * Cluster2D visualization component.
 */
export const Cluster2D: React.FC<Cluster2DProps> = ({
  points,
  embeddings,
  labels,
  colors = DEFAULT_COLORS,
  title,
  xLabel,
  yLabel,
  width,
  height = 400,
  showLegend = true,
  showGrid = true,
  pointSize = 4,
  onPointClick,
  embeddingMethod,
  loading = false
}) => {
  // Prepare data for visualization
  const { clusterData, clusterStats } = useMemo(() => {
    let dataPoints: DataPoint[] = [];
    
    // Handle embedding points
    if (embeddings && embeddings.length > 0) {
      dataPoints = embeddings.map(point => ({
        x: point.x,
        y: point.y,
        cluster: labels ? labels[point.original_index] || 0 : 0,
        index: point.original_index
      }));
    }
    // Handle regular points
    else if (points && points.length > 0) {
      // Validate data
      if (points.some(point => point.length < 2)) {
        console.warn('Some points do not have 2D coordinates');
        return { clusterData: {}, clusterStats: [] };
      }

      // Use cluster labels or default to single cluster
      const effectiveLabels = labels || points.map(() => 0);
      
      if (effectiveLabels.length !== points.length) {
        console.warn('Labels length does not match points length');
        return { clusterData: {}, clusterStats: [] };
      }
      
      dataPoints = points.map((point, index) => ({
        x: point[0],
        y: point[1],
        cluster: effectiveLabels[index],
        index
      }));
    }
    
    if (dataPoints.length === 0) {
      return { clusterData: {}, clusterStats: [] };
    }

    // Group by cluster
    const clustered = dataPoints.reduce((acc, point) => {
      const clusterId = point.cluster;
      if (!acc[clusterId]) {
        acc[clusterId] = [];
      }
      acc[clusterId].push(point);
      return acc;
    }, {} as Record<number, DataPoint[]>);

    // Generate statistics
    const stats = generateClusterStats(dataPoints.map(p => p.cluster));

    return { clusterData: clustered, clusterStats: stats };
  }, [points, embeddings, labels]);

  // Generate dynamic title and labels based on embedding method
  const effectiveTitle = useMemo(() => {
    if (title) return title;
    if (embeddingMethod) {
      return `2D Visualization - ${embeddingMethod.toUpperCase()}`;
    }
    return "2D Cluster Visualization";
  }, [title, embeddingMethod]);

  const effectiveXLabel = useMemo(() => {
    if (xLabel) return xLabel;
    if (embeddingMethod) {
      return `${embeddingMethod.toUpperCase()} Component 1`;
    }
    return "X";
  }, [xLabel, embeddingMethod]);

  const effectiveYLabel = useMemo(() => {
    if (yLabel) return yLabel;
    if (embeddingMethod) {
      return `${embeddingMethod.toUpperCase()} Component 2`;
    }
    return "Y";
  }, [yLabel, embeddingMethod]);

  // Handle loading state
  if (loading) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            {effectiveTitle}
          </Typography>
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
            <Typography variant="body2" color="text.secondary">
              Generating {embeddingMethod || 'visualization'}...
            </Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  // Handle empty or invalid data
  if (Object.keys(clusterData).length === 0) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            {effectiveTitle}
          </Typography>
          <Alert severity="info">
            No data available for visualization.
            {embeddings ? " Please generate embeddings first." : " Please provide valid 2D points."}
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">
            {effectiveTitle}
          </Typography>
          <Stack direction="row" spacing={1}>
            {clusterStats.map(stat => (
              <Chip
                key={stat.clusterId}
                label={`Cluster ${stat.clusterId}: ${stat.count} (${stat.percentage}%)`}
                size="small"
                style={{ 
                  backgroundColor: colors[stat.clusterId % colors.length], 
                  color: 'white' 
                }}
              />
            ))}
          </Stack>
        </Box>

        {/* Chart */}
        <ResponsiveContainer width={width || '100%'} height={height}>
          <ScatterChart margin={{ top: 20, right: 20, bottom: 40, left: 20 }}>
            {showGrid && <CartesianGrid strokeDasharray="3 3" />}
            <XAxis 
              type="number" 
              dataKey="x" 
              name={effectiveXLabel}
              label={{ value: effectiveXLabel, position: 'insideBottom', offset: -5 }}
            />
            <YAxis 
              type="number" 
              dataKey="y" 
              name={effectiveYLabel}
              label={{ value: effectiveYLabel, angle: -90, position: 'insideLeft' }}
            />
            <Tooltip content={<CustomTooltip />} />
            {showLegend && <Legend />}
            
            {Object.entries(clusterData).map(([clusterId, points]) => (
              <Scatter
                key={clusterId}
                name={`Cluster ${clusterId}`}
                data={points}
                fill={colors[parseInt(clusterId) % colors.length]}
                onClick={onPointClick}
                r={pointSize}
              />
            ))}
          </ScatterChart>
        </ResponsiveContainer>

        {/* Footer Stats */}
        <Box sx={{ mt: 2, p: 1, bgcolor: 'background.default', borderRadius: 1 }}>
          <Typography variant="caption" color="text.secondary">
            {clusterStats.reduce((sum, stat) => sum + stat.count, 0)} total points • {clusterStats.length} clusters
            {embeddingMethod && ` • Method: ${embeddingMethod.toUpperCase()}`}
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default Cluster2D;
