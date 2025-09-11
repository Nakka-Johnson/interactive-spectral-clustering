/**
 * DatasetProfile.tsx
 * 
 * Comprehensive dataset statistics and analysis panel component.
 * Displays dataset quality metrics, distributions, and preprocessing recommendations.
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  LinearProgress,
  Alert,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Button,
  CircularProgress
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Assessment as AssessmentIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  TrendingUp as TrendingUpIcon,
  DataUsage as DataUsageIcon,
  Memory as MemoryIcon,
  Speed as SpeedIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';

interface NumericalStats {
  count: number;
  mean: number;
  std: number;
  min: number;
  '25%': number;
  '50%': number;
  '75%': number;
  max: number;
  skewness: number;
  kurtosis: number;
}

interface CategoricalStats {
  unique_count: number;
  top_value: string;
  top_frequency: number;
  top_percentage: number;
}

interface DatasetStats {
  shape: [number, number];
  memory_usage_mb: number;
  dtypes: Record<string, string>;
  missing_counts: Record<string, number>;
  missing_percentages: Record<string, number>;
  total_missing: number;
  numerical_stats: Record<string, NumericalStats>;
  correlations: Record<string, number> | null;
  categorical_stats: Record<string, CategoricalStats>;
  duplicate_rows: number;
  constant_columns: string[];
  high_cardinality_columns: string[];
  skewed_columns: string[];
  outlier_counts: Record<string, number>;
  preprocessing_recommendations: string[];
}

interface DatasetProfileProps {
  datasetId: string;
  onStatsLoaded?: (stats: DatasetStats) => void;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index, ...other }) => {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`dataset-tabpanel-${index}`}
      aria-labelledby={`dataset-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
};

const DatasetProfile: React.FC<DatasetProfileProps> = ({ datasetId, onStatsLoaded }) => {
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<number>(0);

  const fetchStats = React.useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      const token = localStorage.getItem('token');
      const response = await fetch(`/api/datasets/${datasetId}/stats`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch dataset statistics: ${response.statusText}`);
      }

      const data = await response.json();
      setStats(data);
      onStatsLoaded?.(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
    } finally {
      setLoading(false);
    }
  }, [datasetId, onStatsLoaded]);

  useEffect(() => {
    if (datasetId) {
      fetchStats();
    }
  }, [datasetId, fetchStats]);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight={200}>
        <CircularProgress />
        <Typography variant="body1" sx={{ ml: 2 }}>
          Analyzing dataset...
        </Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" action={
        <Button color="inherit" size="small" onClick={fetchStats} startIcon={<RefreshIcon />}>
          Retry
        </Button>
      }>
        {error}
      </Alert>
    );
  }

  if (!stats) {
    return (
      <Alert severity="info">
        No dataset statistics available.
      </Alert>
    );
  }

  const getQualityScore = () => {
    let score = 100;
    
    // Deduct for missing values
    const missingRate = stats.total_missing / (stats.shape[0] * stats.shape[1]);
    score -= missingRate * 30;
    
    // Deduct for duplicates
    const duplicateRate = stats.duplicate_rows / stats.shape[0];
    score -= duplicateRate * 20;
    
    // Deduct for constant columns
    score -= (stats.constant_columns.length / stats.shape[1]) * 15;
    
    // Deduct for high cardinality
    score -= (stats.high_cardinality_columns.length / stats.shape[1]) * 10;
    
    return Math.max(0, Math.round(score));
  };

  const getQualityColor = (score: number) => {
    if (score >= 80) return 'success';
    if (score >= 60) return 'warning';
    return 'error';
  };

  const qualityScore = getQualityScore();

  return (
    <Paper elevation={2} sx={{ p: 2 }}>
      <Box display="flex" justifyContent="between" alignItems="center" mb={2}>
        <Typography variant="h6" gutterBottom>
          Dataset Analysis
        </Typography>
        <Button
          variant="outlined"
          size="small"
          onClick={fetchStats}
          startIcon={<RefreshIcon />}
        >
          Refresh
        </Button>
      </Box>

      {/* Quick Overview Cards */}
      <Box display="grid" gridTemplateColumns="repeat(auto-fit, minmax(200px, 1fr))" gap={3} mb={3}>
        <Card>
          <CardContent>
            <Box display="flex" alignItems="center" mb={1}>
              <DataUsageIcon color="primary" />
              <Typography variant="h6" sx={{ ml: 1 }}>
                Shape
              </Typography>
            </Box>
            <Typography variant="h4">
              {stats.shape[0].toLocaleString()}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              rows × {stats.shape[1]} columns
            </Typography>
          </CardContent>
        </Card>

        <Card>
          <CardContent>
            <Box display="flex" alignItems="center" mb={1}>
              <MemoryIcon color="secondary" />
              <Typography variant="h6" sx={{ ml: 1 }}>
                Memory
              </Typography>
            </Box>
            <Typography variant="h4">
              {stats.memory_usage_mb.toFixed(2)}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              MB
            </Typography>
          </CardContent>
        </Card>

        <Card>
          <CardContent>
            <Box display="flex" alignItems="center" mb={1}>
              <ErrorIcon color="error" />
              <Typography variant="h6" sx={{ ml: 1 }}>
                Missing
              </Typography>
            </Box>
            <Typography variant="h4">
              {stats.total_missing.toLocaleString()}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {((stats.total_missing / (stats.shape[0] * stats.shape[1])) * 100).toFixed(1)}% of total
            </Typography>
          </CardContent>
        </Card>

        <Card>
          <CardContent>
            <Box display="flex" alignItems="center" mb={1}>
              <SpeedIcon color={getQualityColor(qualityScore)} />
              <Typography variant="h6" sx={{ ml: 1 }}>
                Quality Score
              </Typography>
            </Box>
            <Typography variant="h4">
              {qualityScore}
            </Typography>
            <LinearProgress
              variant="determinate"
              value={qualityScore}
              color={getQualityColor(qualityScore)}
              sx={{ mt: 1 }}
            />
          </CardContent>
        </Card>
      </Box>

      {/* Recommendations Alert */}
      {stats.preprocessing_recommendations.length > 0 && (
        <Alert severity="info" sx={{ mb: 3 }}>
          <Typography variant="subtitle2" gutterBottom>
            Preprocessing Recommendations:
          </Typography>
          <List dense>
            {stats.preprocessing_recommendations.map((rec, index) => (
              <ListItem key={index} sx={{ py: 0 }}>
                <ListItemIcon sx={{ minWidth: 30 }}>
                  <InfoIcon fontSize="small" />
                </ListItemIcon>
                <ListItemText primary={rec} />
              </ListItem>
            ))}
          </List>
        </Alert>
      )}

      {/* Detailed Tabs */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={activeTab} onChange={handleTabChange} aria-label="dataset analysis tabs">
          <Tab label="Overview" />
          <Tab label="Missing Values" />
          <Tab label="Statistics" />
          <Tab label="Data Quality" />
          <Tab label="Correlations" />
        </Tabs>
      </Box>

      <TabPanel value={activeTab} index={0}>
        {/* Overview Tab */}
        <Box display="grid" gridTemplateColumns="repeat(auto-fit, minmax(300px, 1fr))" gap={3}>
          <Box>
            <Typography variant="h6" gutterBottom>
              Data Types
            </Typography>
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Column</TableCell>
                    <TableCell>Type</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {Object.entries(stats.dtypes).slice(0, 10).map(([col, dtype]) => (
                    <TableRow key={col}>
                      <TableCell>{col}</TableCell>
                      <TableCell>
                        <Chip label={dtype} size="small" variant="outlined" />
                      </TableCell>
                    </TableRow>
                  ))}
                  {Object.keys(stats.dtypes).length > 10 && (
                    <TableRow>
                      <TableCell colSpan={2} align="center">
                        <Typography variant="body2" color="text.secondary">
                          ... and {Object.keys(stats.dtypes).length - 10} more columns
                        </Typography>
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </TableContainer>
          </Box>

          <Box>
            <Typography variant="h6" gutterBottom>
              Quick Facts
            </Typography>
            <List>
              <ListItem>
                <ListItemIcon>
                  <AssessmentIcon />
                </ListItemIcon>
                <ListItemText
                  primary="Duplicate Rows"
                  secondary={`${stats.duplicate_rows} (${((stats.duplicate_rows / stats.shape[0]) * 100).toFixed(1)}%)`}
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <WarningIcon />
                </ListItemIcon>
                <ListItemText
                  primary="Constant Columns"
                  secondary={stats.constant_columns.length}
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <TrendingUpIcon />
                </ListItemIcon>
                <ListItemText
                  primary="High Cardinality Columns"
                  secondary={stats.high_cardinality_columns.length}
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <InfoIcon />
                </ListItemIcon>
                <ListItemText
                  primary="Skewed Columns"
                  secondary={stats.skewed_columns.length}
                />
              </ListItem>
            </List>
          </Box>
        </Box>
      </TabPanel>

      <TabPanel value={activeTab} index={1}>
        {/* Missing Values Tab */}
        <Typography variant="h6" gutterBottom>
          Missing Values Analysis
        </Typography>
        <TableContainer component={Paper} variant="outlined">
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Column</TableCell>
                <TableCell align="right">Missing Count</TableCell>
                <TableCell align="right">Missing %</TableCell>
                <TableCell>Severity</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {Object.entries(stats.missing_percentages)
                .filter(([_, pct]) => pct > 0)
                .sort(([, a], [, b]) => b - a)
                .map(([col, pct]) => (
                  <TableRow key={col}>
                    <TableCell>{col}</TableCell>
                    <TableCell align="right">{stats.missing_counts[col]}</TableCell>
                    <TableCell align="right">{pct.toFixed(1)}%</TableCell>
                    <TableCell>
                      <Chip
                        label={pct > 50 ? 'High' : pct > 20 ? 'Medium' : 'Low'}
                        color={pct > 50 ? 'error' : pct > 20 ? 'warning' : 'success'}
                        size="small"
                      />
                    </TableCell>
                  </TableRow>
                ))}
            </TableBody>
          </Table>
        </TableContainer>
      </TabPanel>

      <TabPanel value={activeTab} index={2}>
        {/* Statistics Tab */}
        <Typography variant="h6" gutterBottom>
          Numerical Statistics
        </Typography>
        <TableContainer component={Paper} variant="outlined">
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Column</TableCell>
                <TableCell align="right">Mean</TableCell>
                <TableCell align="right">Std</TableCell>
                <TableCell align="right">Min</TableCell>
                <TableCell align="right">Max</TableCell>
                <TableCell align="right">Skewness</TableCell>
                <TableCell align="right">Outliers</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {Object.entries(stats.numerical_stats).map(([col, stat]) => (
                <TableRow key={col}>
                  <TableCell>{col}</TableCell>
                  <TableCell align="right">{stat.mean.toFixed(3)}</TableCell>
                  <TableCell align="right">{stat.std.toFixed(3)}</TableCell>
                  <TableCell align="right">{stat.min.toFixed(3)}</TableCell>
                  <TableCell align="right">{stat.max.toFixed(3)}</TableCell>
                  <TableCell align="right">
                    <Chip
                      label={stat.skewness.toFixed(2)}
                      color={Math.abs(stat.skewness) > 2 ? 'error' : Math.abs(stat.skewness) > 1 ? 'warning' : 'success'}
                      size="small"
                    />
                  </TableCell>
                  <TableCell align="right">{stats.outlier_counts[col] || 0}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </TabPanel>

      <TabPanel value={activeTab} index={3}>
        {/* Data Quality Tab */}
        <Box display="grid" gridTemplateColumns="repeat(auto-fit, minmax(300px, 1fr))" gap={3}>
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="h6">Constant Columns</Typography>
            </AccordionSummary>
            <AccordionDetails>
              {stats.constant_columns.length > 0 ? (
                <List dense>
                  {stats.constant_columns.map((col) => (
                    <ListItem key={col}>
                      <ListItemText primary={col} />
                    </ListItem>
                  ))}
                </List>
              ) : (
                <Typography color="text.secondary">None found</Typography>
              )}
            </AccordionDetails>
          </Accordion>

          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="h6">High Cardinality</Typography>
            </AccordionSummary>
            <AccordionDetails>
              {stats.high_cardinality_columns.length > 0 ? (
                <List dense>
                  {stats.high_cardinality_columns.map((col) => (
                    <ListItem key={col}>
                      <ListItemText primary={col} />
                    </ListItem>
                  ))}
                </List>
              ) : (
                <Typography color="text.secondary">None found</Typography>
              )}
            </AccordionDetails>
          </Accordion>

          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="h6">Skewed Columns</Typography>
            </AccordionSummary>
            <AccordionDetails>
              {stats.skewed_columns.length > 0 ? (
                <List dense>
                  {stats.skewed_columns.map((col) => (
                    <ListItem key={col}>
                      <ListItemText 
                        primary={col}
                        secondary={`Skewness: ${stats.numerical_stats[col]?.skewness.toFixed(2)}`}
                      />
                    </ListItem>
                  ))}
                </List>
              ) : (
                <Typography color="text.secondary">None found</Typography>
              )}
            </AccordionDetails>
          </Accordion>
        </Box>
      </TabPanel>

      <TabPanel value={activeTab} index={4}>
        {/* Correlations Tab */}
        <Typography variant="h6" gutterBottom>
          High Correlations ({"|r| > 0.9"})
        </Typography>
        {stats.correlations && Object.keys(stats.correlations).length > 0 ? (
          <TableContainer component={Paper} variant="outlined">
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Feature Pair</TableCell>
                  <TableCell align="right">Correlation</TableCell>
                  <TableCell>Strength</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {Object.entries(stats.correlations).map(([pair, corr]) => (
                  <TableRow key={pair}>
                    <TableCell>{pair}</TableCell>
                    <TableCell align="right">{corr.toFixed(3)}</TableCell>
                    <TableCell>
                      <Chip
                        label={Math.abs(corr) > 0.95 ? 'Very High' : 'High'}
                        color={Math.abs(corr) > 0.95 ? 'error' : 'warning'}
                        size="small"
                      />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        ) : (
          <Alert severity="success">
            No high correlations found. Features appear to be independent.
          </Alert>
        )}
      </TabPanel>
    </Paper>
  );
};

export default DatasetProfile;
