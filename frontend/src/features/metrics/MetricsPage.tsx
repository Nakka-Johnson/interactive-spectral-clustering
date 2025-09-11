/**
 * MetricsPage component for Interactive Spectral Clustering Platform.
 *
 * Comprehensive metrics dashboard with KPI cards and multiple visualization views.
 * Features algorithm comparison, performance metrics, and detailed analytics.
 */

import React, { useState, useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  ToggleButton,
  ToggleButtonGroup,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Stack,
  Chip,
  Alert,
  Paper,
  Divider,
} from '@mui/material';
import {
  Analytics,
  Assessment,
  TrendingUp,
  Speed,
  EmojiEvents,
  Psychology,
  TableView,
  BarChart,
  Radar,
  Timeline,
} from '@mui/icons-material';
import MetricsComparison, { AlgorithmMetrics } from '../../components/MetricsComparison';

// Mock data for demonstration - in real app this would come from API
const mockMetricsData: AlgorithmMetrics[] = [
  {
    algorithm: 'kmeans',
    silhouette_score: 0.672,
    calinski_harabasz: 156.3,
    davies_bouldin: 0.894,
    execution_time: 0.045,
    run_id: 'run_km_001',
    timestamp: '2025-09-11T10:30:00Z',
  },
  {
    algorithm: 'spectral',
    silhouette_score: 0.734,
    calinski_harabasz: 189.7,
    davies_bouldin: 0.756,
    execution_time: 0.123,
    run_id: 'run_sp_001',
    timestamp: '2025-09-11T10:45:00Z',
  },
  {
    algorithm: 'dbscan',
    silhouette_score: 0.581,
    calinski_harabasz: 134.2,
    davies_bouldin: 1.124,
    execution_time: 0.087,
    run_id: 'run_db_001',
    timestamp: '2025-09-11T11:00:00Z',
  },
  {
    algorithm: 'kmeans',
    silhouette_score: 0.698,
    calinski_harabasz: 167.1,
    davies_bouldin: 0.832,
    execution_time: 0.052,
    run_id: 'run_km_002',
    timestamp: '2025-09-11T11:15:00Z',
  },
  {
    algorithm: 'spectral',
    silhouette_score: 0.712,
    calinski_harabasz: 174.3,
    davies_bouldin: 0.789,
    execution_time: 0.134,
    run_id: 'run_sp_002',
    timestamp: '2025-09-11T11:30:00Z',
  },
];

/**
 * KPI Card Component
 */
interface KPICardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: React.ReactNode;
  color?: 'primary' | 'secondary' | 'success' | 'warning' | 'error';
  trend?: {
    direction: 'up' | 'down' | 'neutral';
    value: string;
  };
}

const KPICard: React.FC<KPICardProps> = ({
  title,
  value,
  subtitle,
  icon,
  color = 'primary',
  trend,
}) => {
  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Stack spacing={2}>
          <Box display="flex" alignItems="center" justifyContent="space-between">
            <Box sx={{ color: `${color}.main` }}>{icon}</Box>
            {trend && (
              <Chip
                label={trend.value}
                size="small"
                color={trend.direction === 'up' ? 'success' : trend.direction === 'down' ? 'error' : 'default'}
                variant="outlined"
              />
            )}
          </Box>
          <Box>
            <Typography variant="h4" fontWeight="bold" color={`${color}.main`}>
              {value}
            </Typography>
            <Typography variant="h6" color="text.primary" gutterBottom>
              {title}
            </Typography>
            {subtitle && (
              <Typography variant="body2" color="text.secondary">
                {subtitle}
              </Typography>
            )}
          </Box>
        </Stack>
      </CardContent>
    </Card>
  );
};

/**
 * Main MetricsPage component
 */
const MetricsPage: React.FC = () => {
  const [selectedView, setSelectedView] = useState<'table' | 'chart' | 'radar' | 'timeline'>('table');
  const [selectedMetric, setSelectedMetric] = useState<string>('silhouette_score');

  // Calculate KPIs from the metrics data
  const kpis = useMemo(() => {
    if (!mockMetricsData.length) return null;

    // Get unique algorithms tested
    const uniqueAlgorithms = [...new Set(mockMetricsData.map(d => d.algorithm))];
    
    // Find best silhouette score
    const bestSilhouetteEntry = mockMetricsData.reduce((best, current) => 
      current.silhouette_score > best.silhouette_score ? current : best
    );
    
    // Calculate average execution time
    const avgExecutionTime = mockMetricsData.reduce((sum, d) => sum + d.execution_time, 0) / mockMetricsData.length;
    
    return {
      algorithmsCount: uniqueAlgorithms.length,
      bestSilhouette: bestSilhouetteEntry.silhouette_score,
      bestAlgorithm: bestSilhouetteEntry.algorithm.toUpperCase(),
      avgExecutionTime: avgExecutionTime,
    };
  }, []);

  // Get the latest run data for trend calculations
  const latestRuns = useMemo(() => {
    const sorted = [...mockMetricsData].sort((a, b) => 
      new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    );
    return sorted.slice(0, 3);
  }, []);

  const handleViewChange = (
    event: React.MouseEvent<HTMLElement>,
    newView: 'table' | 'chart' | 'radar' | 'timeline' | null,
  ) => {
    if (newView !== null) {
      setSelectedView(newView);
    }
  };

  if (!kpis) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="info">
          No metrics data available. Run some clustering algorithms to see performance metrics.
        </Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" gutterBottom>
          Performance Metrics Dashboard
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Comprehensive analysis of clustering algorithm performance across multiple runs and metrics.
        </Typography>
      </Box>

      {/* KPI Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <KPICard
            title="Algorithms Tested"
            value={kpis.algorithmsCount}
            subtitle="Unique clustering algorithms"
            icon={<Psychology fontSize="large" />}
            color="primary"
            trend={{ direction: 'up', value: '+1 this week' }}
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <KPICard
            title="Best Silhouette Score"
            value={kpis.bestSilhouette.toFixed(3)}
            subtitle="Highest clustering quality achieved"
            icon={<EmojiEvents fontSize="large" />}
            color="success"
            trend={{ direction: 'up', value: '+0.062' }}
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <KPICard
            title="Best Algorithm"
            value={kpis.bestAlgorithm}
            subtitle="Top performing clustering method"
            icon={<Assessment fontSize="large" />}
            color="secondary"
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <KPICard
            title="Avg Execution Time"
            value={`${(kpis.avgExecutionTime * 1000).toFixed(0)}ms`}
            subtitle="Average processing time"
            icon={<Speed fontSize="large" />}
            color="warning"
            trend={{ direction: 'down', value: '-15ms' }}
          />
        </Grid>
      </Grid>

      {/* Latest Runs Summary */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Recent Performance Summary
          </Typography>
          <Grid container spacing={2}>
            {latestRuns.map((run, index) => (
              <Grid item xs={12} md={4} key={run.run_id}>
                <Paper sx={{ p: 2, bgcolor: 'background.default' }}>
                  <Stack spacing={1}>
                    <Box display="flex" justifyContent="space-between" alignItems="center">
                      <Typography variant="subtitle1" fontWeight="medium">
                        {run.algorithm.toUpperCase()}
                      </Typography>
                      <Chip 
                        label={`Run ${index + 1}`} 
                        size="small" 
                        color="primary" 
                        variant="outlined" 
                      />
                    </Box>
                    <Divider />
                    <Box display="flex" justifyContent="space-between">
                      <Typography variant="body2" color="text.secondary">
                        Silhouette:
                      </Typography>
                      <Typography variant="body2" fontWeight="medium">
                        {run.silhouette_score.toFixed(3)}
                      </Typography>
                    </Box>
                    <Box display="flex" justifyContent="space-between">
                      <Typography variant="body2" color="text.secondary">
                        Time:
                      </Typography>
                      <Typography variant="body2" fontWeight="medium">
                        {(run.execution_time * 1000).toFixed(0)}ms
                      </Typography>
                    </Box>
                    <Box display="flex" justifyContent="space-between">
                      <Typography variant="body2" color="text.secondary">
                        C-H Index:
                      </Typography>
                      <Typography variant="body2" fontWeight="medium">
                        {run.calinski_harabasz.toFixed(1)}
                      </Typography>
                    </Box>
                  </Stack>
                </Paper>
              </Grid>
            ))}
          </Grid>
        </CardContent>
      </Card>

      {/* Metrics Comparison Controls */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Grid container spacing={3} alignItems="center">
            <Grid item xs={12} md={6}>
              <Box display="flex" alignItems="center" gap={2}>
                <Analytics color="primary" />
                <Typography variant="h6">
                  Detailed Metrics Analysis
                </Typography>
              </Box>
            </Grid>
            
            <Grid item xs={12} md={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Metric for Charts</InputLabel>
                <Select
                  value={selectedMetric}
                  label="Metric for Charts"
                  onChange={(e) => setSelectedMetric(e.target.value)}
                >
                  <MenuItem value="silhouette_score">Silhouette Score</MenuItem>
                  <MenuItem value="calinski_harabasz">Calinski-Harabasz Index</MenuItem>
                  <MenuItem value="davies_bouldin">Davies-Bouldin Index</MenuItem>
                  <MenuItem value="execution_time">Execution Time</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={3}>
              <ToggleButtonGroup
                value={selectedView}
                exclusive
                onChange={handleViewChange}
                size="small"
                fullWidth
              >
                <ToggleButton value="table" aria-label="table view">
                  <TableView fontSize="small" />
                </ToggleButton>
                <ToggleButton value="chart" aria-label="chart view">
                  <BarChart fontSize="small" />
                </ToggleButton>
                <ToggleButton value="radar" aria-label="radar view">
                  <Radar fontSize="small" />
                </ToggleButton>
                <ToggleButton value="timeline" aria-label="timeline view">
                  <Timeline fontSize="small" />
                </ToggleButton>
              </ToggleButtonGroup>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Metrics Comparison Views */}
      <MetricsComparison
        data={mockMetricsData}
        selectedMetric={selectedMetric}
        view={selectedView}
      />

      {/* Performance Insights */}
      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Performance Insights
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} md={4}>
              <Alert severity="success" sx={{ height: '100%' }}>
                <Typography variant="subtitle2" gutterBottom>
                  <strong>Best Overall:</strong> Spectral Clustering
                </Typography>
                <Typography variant="body2">
                  Achieved highest average silhouette score (0.723) with good cluster separation.
                </Typography>
              </Alert>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Alert severity="info" sx={{ height: '100%' }}>
                <Typography variant="subtitle2" gutterBottom>
                  <strong>Fastest Algorithm:</strong> K-Means
                </Typography>
                <Typography variant="body2">
                  Consistently delivers results in under 60ms, ideal for real-time applications.
                </Typography>
              </Alert>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Alert severity="warning" sx={{ height: '100%' }}>
                <Typography variant="subtitle2" gutterBottom>
                  <strong>Room for Improvement:</strong> DBSCAN
                </Typography>
                <Typography variant="body2">
                  Consider parameter tuning to improve silhouette scores and reduce noise points.
                </Typography>
              </Alert>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </Box>
  );
};

export default MetricsPage;
