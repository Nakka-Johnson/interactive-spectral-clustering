/**
 * RunProgress component for Interactive Spectral Clustering Platform.
 * Shows real-time progress updates and logs for clustering runs.
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  LinearProgress,
  Chip,
  List,
  ListItem,
  ListItemText,
  IconButton,
  Collapse,
  Alert,
  Button
} from '@mui/material';
import {
  ExpandMore,
  ExpandLess,
  Refresh,
  PlayArrow,
  CheckCircle,
  Error,
  Schedule
} from '@mui/icons-material';
import { useWebSocket, ProgressUpdate, CompletionData, ErrorData } from '../../lib/ws';
import { httpClient, tokenManager } from '../../lib/http';
import { logger } from '../../utils/logger';

// Helper for shallow equality comparison
const shallowEqualObj = (a: any, b: any) => {
  if (Object.is(a, b)) return true;
  if (!a || !b || typeof a !== 'object' || typeof b !== 'object') return false;
  const ka = Object.keys(a), kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  for (const k of ka) if (!Object.is((a as any)[k], (b as any)[k])) return false;
  return true;
};

interface RunProgressProps {
  runId: string;
  onComplete?: (result: any) => void;
  onError?: (error: string) => void;
  showLogs?: boolean;
  autoCollapse?: boolean;
}

interface LogEntry {
  timestamp: string;
  message: string;
  progress: number;
  status: string;
}

export const RunProgress: React.FC<RunProgressProps> = ({
  runId,
  onComplete,
  onError,
  showLogs = true,
  autoCollapse = false
}) => {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isLogsExpanded, setIsLogsExpanded] = useState(!autoCollapse);
  const [currentProgress, setCurrentProgress] = useState(0);
  const [currentStatus, setCurrentStatus] = useState<string>('queued');
  const [currentMessage, setCurrentMessage] = useState('Initializing...');
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [usePolling, setUsePolling] = useState(false);
  const [pollingInterval, setPollingInterval] = useState<NodeJS.Timeout | null>(null);

  const token = tokenManager.getToken();

  // Memoized callback handlers
  const handleProgress = useCallback((data: ProgressUpdate) => {
    setCurrentProgress(data.progress);
    setCurrentStatus(data.status);
    setCurrentMessage(data.message);
    setLastUpdate(new Date());
    
    // Add to logs
    const logEntry: LogEntry = {
      timestamp: data.timestamp,
      message: data.message,
      progress: data.progress,
      status: data.status
    };
    setLogs(prev => [...prev, logEntry]);
  }, []);

  const handleCompletion = useCallback((data: CompletionData) => {
    setCurrentProgress(100);
    setCurrentStatus('done');
    setCurrentMessage('Clustering completed successfully!');
    setLastUpdate(new Date());
    
    const logEntry: LogEntry = {
      timestamp: data.timestamp,
      message: data.message,
      progress: data.progress,
      status: data.status
    };
    setLogs(prev => [...prev, logEntry]);
    
    onComplete?.(data.result);
  }, [onComplete]);

  const handleError = useCallback((data: ErrorData) => {
    setCurrentProgress(0);
    setCurrentStatus('error');
    setCurrentMessage(data.error);
    setLastUpdate(new Date());
    
    const logEntry: LogEntry = {
      timestamp: data.timestamp,
      message: data.error,
      progress: data.progress,
      status: data.status
    };
    setLogs(prev => [...prev, logEntry]);
    
    onError?.(data.error);
  }, [onError]);

  const handleConnectionChange = useCallback((connected: boolean) => {
    if (!connected && currentStatus === 'running') {
      // Switch to polling if WebSocket fails during active run
      logger.info('[RunProgress] WebSocket disconnected, switching to polling');
      setUsePolling(true);
    }
  }, [currentStatus]);

  // WebSocket connection with fallback to polling
  const {
    isConnected,
  } = useWebSocket(runId, token, {
    enabled: !usePolling,
    onProgress: handleProgress,
    onCompletion: handleCompletion,
    onError: handleError,
    onConnectionChange: handleConnectionChange
  });

  // Polling fallback - fixed to avoid infinite loops
  useEffect(() => {
    if (!usePolling) return;
    
    const poll = async () => {
      try {
        const response = await httpClient.get(`/runs/${runId}`);
        const data = response.data;
        
        setCurrentProgress(prev => prev === data.progress ? prev : data.progress);
        setCurrentStatus(prev => prev === data.status ? prev : data.status);
        setCurrentMessage(prev => prev === data.message ? prev : data.message);
        setLastUpdate(new Date());
        
        // Add to logs if message changed
        setLogs(prev => {
          if (prev.length === 0 || prev[prev.length - 1].message !== data.message) {
            const logEntry: LogEntry = {
              timestamp: data.timestamp,
              message: data.message,
              progress: data.progress,
              status: data.status
            };
            const newLogs = [...prev, logEntry];
            return shallowEqualObj(prev, newLogs) ? prev : newLogs;
          }
          return prev;
        });
        
        // Handle completion
        if (data.status === 'done' && data.metrics) {
          onComplete?.(data.metrics);
        } else if (data.status === 'error') {
          onError?.(data.error_message);
        }
        
      } catch (error) {
        console.error('[RunProgress] Polling error:', error);
      }
    };

    // Start interval polling - only depend on usePolling and runId
    const interval = setInterval(poll, 2000); // Poll every 2 seconds
    setPollingInterval(interval);

    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [usePolling, runId]); // ✅ removed currentStatus to prevent infinite loops

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollingInterval) {
        clearInterval(pollingInterval);
      }
    };
  }, [pollingInterval]);

  const getStatusIcon = () => {
    switch (currentStatus) {
      case 'queued':
        return <Schedule color="warning" />;
      case 'running':
        return <PlayArrow color="primary" />;
      case 'done':
        return <CheckCircle color="success" />;
      case 'error':
        return <Error color="error" />;
      default:
        return <Schedule />;
    }
  };

  const getStatusColor = () => {
    switch (currentStatus) {
      case 'queued':
        return 'warning' as const;
      case 'running':
        return 'primary' as const;
      case 'done':
        return 'success' as const;
      case 'error':
        return 'error' as const;
      default:
        return 'default' as const;
    }
  };

  const formatTimestamp = (timestamp: string) => {
    try {
      return new Date(timestamp).toLocaleTimeString();
    } catch {
      return timestamp;
    }
  };

  const retryConnection = () => {
    setUsePolling(false);
    setLogs([]);
    setCurrentProgress(0);
    setCurrentStatus('queued');
    setCurrentMessage('Reconnecting...');
  };

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          {getStatusIcon()}
          <Typography variant="h6" sx={{ ml: 1, flexGrow: 1 }}>
            Run Progress
          </Typography>
          <Chip 
            label={currentStatus.toUpperCase()} 
            color={getStatusColor()}
            size="small"
          />
        </Box>

        {/* Connection Status */}
        {usePolling && (
          <Alert 
            severity="info" 
            sx={{ mb: 2 }}
            action={
              <Button size="small" onClick={retryConnection} startIcon={<Refresh />}>
                Retry WebSocket
              </Button>
            }
          >
            Using polling fallback (WebSocket unavailable)
          </Alert>
        )}

        {/* Progress Bar */}
        <Box sx={{ mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
            <Typography variant="body2" sx={{ flexGrow: 1 }}>
              {currentMessage}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {currentProgress}%
            </Typography>
          </Box>
          <LinearProgress 
            variant="determinate" 
            value={currentProgress} 
            color={currentStatus === 'error' ? 'error' : 'primary'}
          />
        </Box>

        {/* Last Update */}
        {lastUpdate && (
          <Typography variant="caption" color="text.secondary" sx={{ mb: 2, display: 'block' }}>
            Last update: {lastUpdate.toLocaleTimeString()}
            {!isConnected && !usePolling && ' (disconnected)'}
          </Typography>
        )}

        {/* Logs Section */}
        {showLogs && logs.length > 0 && (
          <Box>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <Typography variant="subtitle2" sx={{ flexGrow: 1 }}>
                Progress Log ({logs.length} entries)
              </Typography>
              <IconButton
                size="small"
                onClick={() => setIsLogsExpanded(!isLogsExpanded)}
              >
                {isLogsExpanded ? <ExpandLess /> : <ExpandMore />}
              </IconButton>
            </Box>

            <Collapse in={isLogsExpanded}>
              <List dense sx={{ maxHeight: 200, overflow: 'auto', bgcolor: 'grey.50', borderRadius: 1 }}>
                {logs.map((log, index) => (
                  <ListItem key={index} divider={index < logs.length - 1}>
                    <ListItemText
                      primary={log.message}
                      secondary={
                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span>{formatTimestamp(log.timestamp)}</span>
                          <span>{log.progress}%</span>
                        </Box>
                      }
                    />
                  </ListItem>
                ))}
              </List>
            </Collapse>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default RunProgress;
