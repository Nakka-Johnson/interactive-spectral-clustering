/**
 * MetricsComparison component for Interactive Spectral Clustering Platform.
 *
 * Comprehensive metrics comparison component with multiple visualization views:
 * Table, Chart, Radar, and Timeline views for algorithm performance analysis.
 */

import React, { useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Paper,
  useTheme,
} from '@mui/material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
  AreaChart,
  Area,
} from 'recharts';

// Types for metrics data
export interface AlgorithmMetrics {
  algorithm: string;
  silhouette_score: number;
  calinski_harabasz: number;
  davies_bouldin: number;
  execution_time: number;
  run_id: string;
  timestamp: string;
}

export interface MetricsComparisonProps {
  data: AlgorithmMetrics[];
  selectedMetric: string;
  view: 'table' | 'chart' | 'radar' | 'timeline';
}

/**
 * Get color for score based on quality (higher is better for most metrics)
 */
const getScoreColor = (
  metric: string,
  value: number
): 'success' | 'warning' | 'error' => {
  if (metric === 'silhouette_score') {
    if (value >= 0.7) return 'success';
    if (value >= 0.5) return 'warning';
    return 'error';
  }

  if (metric === 'calinski_harabasz') {
    if (value >= 150) return 'success';
    if (value >= 100) return 'warning';
    return 'error';
  }

  if (metric === 'davies_bouldin') {
    // Lower is better for Davies-Bouldin
    if (value <= 0.8) return 'success';
    if (value <= 1.2) return 'warning';
    return 'error';
  }

  if (metric === 'execution_time') {
    // Lower is better for execution time
    if (value <= 0.05) return 'success';
    if (value <= 0.15) return 'warning';
    return 'error';
  }

  return 'warning';
};

/**
 * Format metric values for display
 */
const formatMetricValue = (metric: string, value: number): string => {
  if (metric === 'execution_time') {
    return `${(value * 1000).toFixed(1)}ms`;
  }
  if (metric === 'calinski_harabasz') {
    return value.toFixed(1);
  }
  return value.toFixed(3);
};

/**
 * Get metric display name
 */
const getMetricDisplayName = (metric: string): string => {
  const names = {
    silhouette_score: 'Silhouette Score',
    calinski_harabasz: 'Calinski-Harabasz Index',
    davies_bouldin: 'Davies-Bouldin Index',
    execution_time: 'Execution Time',
  };
  return names[metric as keyof typeof names] || metric;
};

/**
 * Table View Component
 */
const TableView: React.FC<{ data: AlgorithmMetrics[] }> = ({ data }) => {
  return (
    <TableContainer component={Paper} sx={{ maxHeight: 400 }}>
      <Table stickyHeader>
        <TableHead>
          <TableRow>
            <TableCell>Algorithm</TableCell>
            <TableCell align="center">Silhouette Score</TableCell>
            <TableCell align="center">Calinski-Harabasz</TableCell>
            <TableCell align="center">Davies-Bouldin</TableCell>
            <TableCell align="center">Execution Time</TableCell>
            <TableCell align="center">Run ID</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {data.map((row, index) => (
            <TableRow key={index} hover>
              <TableCell component="th" scope="row">
                <Typography variant="subtitle2" fontWeight="medium">
                  {row.algorithm.toUpperCase()}
                </Typography>
              </TableCell>
              <TableCell align="center">
                <Chip
                  label={formatMetricValue('silhouette_score', row.silhouette_score)}
                  color={getScoreColor('silhouette_score', row.silhouette_score)}
                  variant="outlined"
                  size="small"
                />
              </TableCell>
              <TableCell align="center">
                <Chip
                  label={formatMetricValue('calinski_harabasz', row.calinski_harabasz)}
                  color={getScoreColor('calinski_harabasz', row.calinski_harabasz)}
                  variant="outlined"
                  size="small"
                />
              </TableCell>
              <TableCell align="center">
                <Chip
                  label={formatMetricValue('davies_bouldin', row.davies_bouldin)}
                  color={getScoreColor('davies_bouldin', row.davies_bouldin)}
                  variant="outlined"
                  size="small"
                />
              </TableCell>
              <TableCell align="center">
                <Chip
                  label={formatMetricValue('execution_time', row.execution_time)}
                  color={getScoreColor('execution_time', row.execution_time)}
                  variant="outlined"
                  size="small"
                />
              </TableCell>
              <TableCell align="center">
                <Typography variant="body2" color="text.secondary">
                  {row.run_id.substring(0, 8)}...
                </Typography>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

/**
 * Chart View Component
 */
const ChartView: React.FC<{
  data: AlgorithmMetrics[];
  selectedMetric: string;
}> = ({ data, selectedMetric }) => {
  const theme = useTheme();

  const chartData = useMemo(() => {
    return data.map((item) => ({
      algorithm: item.algorithm.toUpperCase(),
      value: item[selectedMetric as keyof AlgorithmMetrics] as number,
      displayValue: formatMetricValue(
        selectedMetric,
        item[selectedMetric as keyof AlgorithmMetrics] as number
      ),
    }));
  }, [data, selectedMetric]);

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        {getMetricDisplayName(selectedMetric)} Comparison
      </Typography>
      <ResponsiveContainer width="100%" height={350}>
        <BarChart
          data={chartData}
          margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke={theme.palette.divider}
            opacity={0.3}
          />
          <XAxis
            dataKey="algorithm"
            stroke={theme.palette.text.secondary}
            fontSize={12}
          />
          <YAxis stroke={theme.palette.text.secondary} fontSize={12} />
          <Tooltip
            contentStyle={{
              backgroundColor: theme.palette.background.paper,
              border: `1px solid ${theme.palette.divider}`,
              borderRadius: 8,
              color: theme.palette.text.primary,
            }}
            formatter={(value: any, name: string) => [
              formatMetricValue(selectedMetric, value),
              getMetricDisplayName(selectedMetric),
            ]}
          />
          <Bar
            dataKey="value"
            fill={theme.palette.primary.main}
            radius={[4, 4, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </Box>
  );
};

/**
 * Radar View Component
 */
const RadarView: React.FC<{ data: AlgorithmMetrics[] }> = ({ data }) => {
  const theme = useTheme();

  const radarData = useMemo(() => {
    // Normalize metrics to 0-100 scale for radar chart
    const metrics = [
      'silhouette_score',
      'calinski_harabasz',
      'davies_bouldin',
      'execution_time',
    ];

    return metrics.map((metric) => {
      const values = data.map(
        (item) => item[metric as keyof AlgorithmMetrics] as number
      );
      const min = Math.min(...values);
      const max = Math.max(...values);

      const point: any = {
        metric: getMetricDisplayName(metric),
      };

      data.forEach((item) => {
        let normalizedValue =
          (((item[metric as keyof AlgorithmMetrics] as number) - min) /
            (max - min)) *
          100;

        // Invert for metrics where lower is better
        if (metric === 'davies_bouldin' || metric === 'execution_time') {
          normalizedValue = 100 - normalizedValue;
        }

        point[item.algorithm.toUpperCase()] = normalizedValue;
      });

      return point;
    });
  }, [data]);

  const algorithmNames = data.map((item) => item.algorithm.toUpperCase());
  const colors = [
    theme.palette.primary.main,
    theme.palette.secondary.main,
    theme.palette.error.main,
  ];

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Multi-Metric Radar Comparison
      </Typography>
      <ResponsiveContainer width="100%" height={350}>
        <RadarChart
          data={radarData}
          margin={{ top: 20, right: 80, bottom: 20, left: 80 }}
        >
          <PolarGrid stroke={theme.palette.divider} opacity={0.3} />
          <PolarAngleAxis
            dataKey="metric"
            tick={{ fill: theme.palette.text.secondary, fontSize: 12 }}
          />
          <PolarRadiusAxis
            angle={90}
            domain={[0, 100]}
            tick={{ fill: theme.palette.text.secondary, fontSize: 10 }}
            tickCount={5}
          />
          {algorithmNames.map((algorithm, index) => (
            <Radar
              key={algorithm}
              name={algorithm}
              dataKey={algorithm}
              stroke={colors[index % colors.length]}
              fill={colors[index % colors.length]}
              fillOpacity={0.1}
              strokeWidth={2}
            />
          ))}
          <Legend
            wrapperStyle={{ color: theme.palette.text.primary }}
            iconType="line"
          />
        </RadarChart>
      </ResponsiveContainer>
      <Typography
        variant="caption"
        color="text.secondary"
        sx={{ mt: 1, display: 'block' }}
      >
        Values normalized to 0-100 scale. Higher values represent better
        performance.
      </Typography>
    </Box>
  );
};

/**
 * Timeline View Component
 */
const TimelineView: React.FC<{
  data: AlgorithmMetrics[];
  selectedMetric: string;
}> = ({ data, selectedMetric }) => {
  const theme = useTheme();

  const timelineData = useMemo(() => {
    // Sort by timestamp and create timeline points
    const sortedData = [...data].sort(
      (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );

    return sortedData.map((item, index) => ({
      timestamp: new Date(item.timestamp).toLocaleDateString(),
      run: index + 1,
      [item.algorithm]: item[selectedMetric as keyof AlgorithmMetrics] as number,
      algorithm: item.algorithm.toUpperCase(),
    }));
  }, [data, selectedMetric]);

  const algorithmNames = [
    ...new Set(data.map((item) => item.algorithm.toUpperCase())),
  ];
  const colors = [
    theme.palette.primary.main,
    theme.palette.secondary.main,
    theme.palette.error.main,
  ];

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        {getMetricDisplayName(selectedMetric)} Timeline
      </Typography>
      <ResponsiveContainer width="100%" height={350}>
        <AreaChart
          data={timelineData}
          margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke={theme.palette.divider}
            opacity={0.3}
          />
          <XAxis
            dataKey="timestamp"
            stroke={theme.palette.text.secondary}
            fontSize={12}
          />
          <YAxis stroke={theme.palette.text.secondary} fontSize={12} />
          <Tooltip
            contentStyle={{
              backgroundColor: theme.palette.background.paper,
              border: `1px solid ${theme.palette.divider}`,
              borderRadius: 8,
              color: theme.palette.text.primary,
            }}
            formatter={(value: any) => [
              formatMetricValue(selectedMetric, value),
              getMetricDisplayName(selectedMetric),
            ]}
          />
          {algorithmNames.map((algorithm, index) => (
            <Area
              key={algorithm}
              type="monotone"
              dataKey={algorithm.toLowerCase()}
              stroke={colors[index % colors.length]}
              fill={colors[index % colors.length]}
              fillOpacity={0.2}
              strokeWidth={2}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
    </Box>
  );
};

/**
 * Main MetricsComparison component
 */
const MetricsComparison: React.FC<MetricsComparisonProps> = ({
  data,
  selectedMetric,
  view,
}) => {
  if (!data || data.length === 0) {
    return (
      <Card>
        <CardContent>
          <Typography color="text.secondary" align="center">
            No metrics data available
          </Typography>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent>
        {view === 'table' && <TableView data={data} />}
        {view === 'chart' && (
          <ChartView data={data} selectedMetric={selectedMetric} />
        )}
        {view === 'radar' && <RadarView data={data} />}
        {view === 'timeline' && (
          <TimelineView data={data} selectedMetric={selectedMetric} />
        )}
      </CardContent>
    </Card>
  );
};

export default MetricsComparison;

