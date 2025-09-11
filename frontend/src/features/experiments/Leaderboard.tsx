import React, { useState, useEffect, useCallback } from 'react';
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
  TableSortLabel,
  Paper,
  Chip,
  IconButton,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Alert,
  LinearProgress,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  Launch as LaunchIcon,
  Refresh as RefreshIcon,
  TrendingUp as TrendingUpIcon,
  Timer as TimerIcon,
  Memory as MemoryIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';

interface LeaderboardEntry {
  rank: number;
  run_id: string;
  experiment_name: string;
  algorithm: string;
  parameters: Record<string, any>;
  optimization_score: number;
  silhouette_score?: number;
  davies_bouldin_score?: number;
  calinski_harabasz_score?: number;
  execution_time: number;
  gpu_used: boolean;
  completed_at: string;
}

interface LeaderboardProps {
  onSelectRun: (runId: string) => void;
  onRefresh: () => void;
}

type SortField = 'rank' | 'optimization_score' | 'execution_time' | 'completed_at';
type SortDirection = 'asc' | 'desc';

export const Leaderboard: React.FC<LeaderboardProps> = ({
  onSelectRun,
  onRefresh,
}) => {
  const [entries, setEntries] = useState<LeaderboardEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [sortField, setSortField] = useState<SortField>('rank');
  const [sortDirection, setSortDirection] = useState<SortDirection>('asc');
  const [selectedMetric, setSelectedMetric] = useState('silhouette_score');
  const [algorithmFilter, setAlgorithmFilter] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedEntry, setSelectedEntry] = useState<LeaderboardEntry | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);

  // Available metrics for filtering
  const metrics = [
    { value: 'silhouette_score', label: 'Silhouette Score', higherBetter: true },
    { value: 'davies_bouldin_score', label: 'Davies-Bouldin Score', higherBetter: false },
    { value: 'calinski_harabasz_score', label: 'Calinski-Harabasz Score', higherBetter: true },
  ];

  // Get unique algorithms from entries
  const algorithms = Array.from(new Set(entries.map(e => e.algorithm)));

  const fetchLeaderboard = useCallback(async () => {
    setLoading(true);
    try {
      // Mock API call - replace with actual API
      const response = await fetch(`/api/leaderboard?metric=${selectedMetric}&limit=100`);
      if (response.ok) {
        const data = await response.json();
        setEntries(data.leaderboard || []);
      }
    } catch (error) {
      console.error('Error fetching leaderboard:', error);
      // Mock data for development
      setEntries(generateMockData());
    } finally {
      setLoading(false);
    }
  }, [selectedMetric]);

  useEffect(() => {
    fetchLeaderboard();
  }, [fetchLeaderboard]);

  const generateMockData = (): LeaderboardEntry[] => {
    const algorithms = ['spectral', 'dbscan', 'gmm', 'agglomerative'];
    const experiments = ['Parameter Tuning #1', 'GPU Performance Test', 'Large Dataset Analysis'];
    
    return Array.from({ length: 20 }, (_, i) => ({
      rank: i + 1,
      run_id: `run_${i + 1}`,
      experiment_name: experiments[i % experiments.length],
      algorithm: algorithms[i % algorithms.length],
      parameters: {
        n_clusters: Math.floor(Math.random() * 8) + 2,
        eps: Math.random() * 2,
        gamma: Math.random() * 5,
      },
      optimization_score: Math.random() * 0.8 + 0.2,
      silhouette_score: Math.random() * 0.8 + 0.2,
      davies_bouldin_score: Math.random() * 2 + 0.5,
      calinski_harabasz_score: Math.random() * 1000 + 100,
      execution_time: Math.random() * 120 + 10,
      gpu_used: Math.random() > 0.5,
      completed_at: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString(),
    }));
  };

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  const filteredAndSortedEntries = React.useMemo(() => {
    let filtered = entries;

    // Apply algorithm filter
    if (algorithmFilter !== 'all') {
      filtered = filtered.filter(entry => entry.algorithm === algorithmFilter);
    }

    // Apply search filter
    if (searchTerm) {
      filtered = filtered.filter(entry =>
        entry.experiment_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        entry.algorithm.toLowerCase().includes(searchTerm.toLowerCase()) ||
        entry.run_id.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Apply sorting
    filtered.sort((a, b) => {
      let aValue: any, bValue: any;
      
      switch (sortField) {
        case 'rank':
          aValue = a.rank;
          bValue = b.rank;
          break;
        case 'optimization_score':
          aValue = a.optimization_score;
          bValue = b.optimization_score;
          break;
        case 'execution_time':
          aValue = a.execution_time;
          bValue = b.execution_time;
          break;
        case 'completed_at':
          aValue = new Date(a.completed_at).getTime();
          bValue = new Date(b.completed_at).getTime();
          break;
        default:
          return 0;
      }

      if (aValue < bValue) return sortDirection === 'asc' ? -1 : 1;
      if (aValue > bValue) return sortDirection === 'asc' ? 1 : -1;
      return 0;
    });

    return filtered;
  }, [entries, algorithmFilter, searchTerm, sortField, sortDirection]);

  const formatScore = (score: number | undefined) => {
    return score ? score.toFixed(3) : 'N/A';
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString() + ' ' + 
           new Date(dateString).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const getScoreColor = (score: number, metric: string) => {
    const metricInfo = metrics.find(m => m.value === metric);
    if (!metricInfo) return 'text.primary';
    
    if (metricInfo.higherBetter) {
      if (score > 0.7) return 'success.main';
      if (score > 0.5) return 'warning.main';
      return 'error.main';
    } else {
      if (score < 1.0) return 'success.main';
      if (score < 2.0) return 'warning.main';
      return 'error.main';
    }
  };

  const openDetails = (entry: LeaderboardEntry) => {
    setSelectedEntry(entry);
    setDetailsOpen(true);
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Experiment Leaderboard
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Top performing clustering runs across all experiments
      </Typography>

      {/* Controls */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr', md: '1fr 1fr 2fr 1fr' }, gap: 2, alignItems: 'center' }}>
            <FormControl fullWidth size="small">
              <InputLabel>Optimization Metric</InputLabel>
              <Select
                value={selectedMetric}
                label="Optimization Metric"
                onChange={(e) => setSelectedMetric(e.target.value)}
              >
                {metrics.map(metric => (
                  <MenuItem key={metric.value} value={metric.value}>
                    {metric.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControl fullWidth size="small">
              <InputLabel>Algorithm</InputLabel>
              <Select
                value={algorithmFilter}
                label="Algorithm"
                onChange={(e) => setAlgorithmFilter(e.target.value)}
              >
                <MenuItem value="all">All Algorithms</MenuItem>
                {algorithms.map(alg => (
                  <MenuItem key={alg} value={alg}>
                    {alg.charAt(0).toUpperCase() + alg.slice(1)}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <TextField
              fullWidth
              size="small"
              label="Search experiments..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />

            <Button
              fullWidth
              variant="outlined"
              onClick={() => {
                onRefresh();
                fetchLeaderboard();
              }}
              startIcon={<RefreshIcon />}
              disabled={loading}
            >
              Refresh
            </Button>
          </Box>
        </CardContent>
      </Card>

      {/* Loading */}
      {loading && <LinearProgress sx={{ mb: 2 }} />}

      {/* Results count */}
      <Box sx={{ mb: 2 }}>
        <Typography variant="body2" color="text.secondary">
          Showing {filteredAndSortedEntries.length} of {entries.length} results
        </Typography>
      </Box>

      {/* Leaderboard Table */}
      <Card>
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>
                  <TableSortLabel
                    active={sortField === 'rank'}
                    direction={sortField === 'rank' ? sortDirection : 'asc'}
                    onClick={() => handleSort('rank')}
                  >
                    Rank
                  </TableSortLabel>
                </TableCell>
                <TableCell>Experiment</TableCell>
                <TableCell>Algorithm</TableCell>
                <TableCell>
                  <TableSortLabel
                    active={sortField === 'optimization_score'}
                    direction={sortField === 'optimization_score' ? sortDirection : 'asc'}
                    onClick={() => handleSort('optimization_score')}
                  >
                    Score
                  </TableSortLabel>
                </TableCell>
                <TableCell>Metrics</TableCell>
                <TableCell>
                  <TableSortLabel
                    active={sortField === 'execution_time'}
                    direction={sortField === 'execution_time' ? sortDirection : 'asc'}
                    onClick={() => handleSort('execution_time')}
                  >
                    Time
                  </TableSortLabel>
                </TableCell>
                <TableCell>GPU</TableCell>
                <TableCell>
                  <TableSortLabel
                    active={sortField === 'completed_at'}
                    direction={sortField === 'completed_at' ? sortDirection : 'asc'}
                    onClick={() => handleSort('completed_at')}
                  >
                    Completed
                  </TableSortLabel>
                </TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {filteredAndSortedEntries.map((entry) => (
                <TableRow key={entry.run_id} hover>
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      {entry.rank <= 3 && (
                        <TrendingUpIcon 
                          sx={{ 
                            mr: 1, 
                            color: entry.rank === 1 ? 'gold' : entry.rank === 2 ? 'silver' : '#CD7F32' 
                          }} 
                        />
                      )}
                      <Typography variant="h6">#{entry.rank}</Typography>
                    </Box>
                  </TableCell>
                  
                  <TableCell>
                    <Box>
                      <Typography variant="body2" fontWeight="medium">
                        {entry.experiment_name}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {entry.run_id}
                      </Typography>
                    </Box>
                  </TableCell>
                  
                  <TableCell>
                    <Chip 
                      label={entry.algorithm} 
                      size="small" 
                      color="primary" 
                      variant="outlined"
                    />
                  </TableCell>
                  
                  <TableCell>
                    <Typography 
                      variant="h6" 
                      sx={{ color: getScoreColor(entry.optimization_score, selectedMetric) }}
                    >
                      {formatScore(entry.optimization_score)}
                    </Typography>
                  </TableCell>
                  
                  <TableCell>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                      <Typography variant="caption">
                        Sil: {formatScore(entry.silhouette_score)}
                      </Typography>
                      <Typography variant="caption">
                        DB: {formatScore(entry.davies_bouldin_score)}
                      </Typography>
                      <Typography variant="caption">
                        CH: {formatScore(entry.calinski_harabasz_score)}
                      </Typography>
                    </Box>
                  </TableCell>
                  
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <TimerIcon sx={{ mr: 0.5, fontSize: 16 }} />
                      <Typography variant="body2">
                        {formatDuration(entry.execution_time)}
                      </Typography>
                    </Box>
                  </TableCell>
                  
                  <TableCell>
                    <Chip
                      icon={<MemoryIcon />}
                      label={entry.gpu_used ? 'GPU' : 'CPU'}
                      size="small"
                      color={entry.gpu_used ? 'success' : 'default'}
                      variant="outlined"
                    />
                  </TableCell>
                  
                  <TableCell>
                    <Typography variant="caption">
                      {formatDate(entry.completed_at)}
                    </Typography>
                  </TableCell>
                  
                  <TableCell>
                    <Box sx={{ display: 'flex', gap: 1 }}>
                      <Tooltip title="View details">
                        <IconButton 
                          size="small" 
                          onClick={() => openDetails(entry)}
                        >
                          <SettingsIcon />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="View results">
                        <IconButton 
                          size="small" 
                          onClick={() => onSelectRun(entry.run_id)}
                        >
                          <LaunchIcon />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Card>

      {filteredAndSortedEntries.length === 0 && !loading && (
        <Alert severity="info" sx={{ mt: 2 }}>
          No experiments found matching your criteria.
        </Alert>
      )}

      {/* Details Dialog */}
      <Dialog 
        open={detailsOpen} 
        onClose={() => setDetailsOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Run Details: {selectedEntry?.run_id}
        </DialogTitle>
        <DialogContent>
          {selectedEntry && (
            <Box sx={{ pt: 1 }}>
              <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 2, mb: 2 }}>
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Experiment
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 2 }}>
                    {selectedEntry.experiment_name}
                  </Typography>
                  
                  <Typography variant="subtitle2" gutterBottom>
                    Algorithm
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 2 }}>
                    {selectedEntry.algorithm}
                  </Typography>
                </Box>
                
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Performance
                  </Typography>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2">
                      Optimization Score: {formatScore(selectedEntry.optimization_score)}
                    </Typography>
                    <Typography variant="body2">
                      Execution Time: {formatDuration(selectedEntry.execution_time)}
                    </Typography>
                    <Typography variant="body2">
                      GPU Used: {selectedEntry.gpu_used ? 'Yes' : 'No'}
                    </Typography>
                  </Box>
                </Box>
              </Box>
                
              <Box>
                <Typography variant="subtitle2" gutterBottom>
                  Parameters
                </Typography>
                <Paper variant="outlined" sx={{ p: 2 }}>
                  <pre style={{ margin: 0, fontSize: '0.875rem' }}>
                    {JSON.stringify(selectedEntry.parameters, null, 2)}
                  </pre>
                </Paper>
              </Box>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDetailsOpen(false)}>Close</Button>
          <Button 
            variant="contained" 
            onClick={() => {
              setDetailsOpen(false);
              if (selectedEntry) {
                onSelectRun(selectedEntry.run_id);
              }
            }}
          >
            View Results
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};
