/**
 * Test page for PHASE 2 WebSocket functionality.
 * Demonstrates real-time progress tracking and polling fallback.
 */

import React, { useState } from 'react';
import {
  Container,
  Typography,
  Button,
  Card,
  CardContent,
  TextField,
  Stack,
  Alert
} from '@mui/material';
import { PlayArrow } from '@mui/icons-material';
import RunProgress from '../features/experiments/RunProgress';
import { httpClient, tokenManager } from '../lib/http';

export const Phase2TestPage: React.FC = () => {
  const [runId, setRunId] = useState<string>('');
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [currentRunId, setCurrentRunId] = useState<string | null>(null);
  const [error, setError] = useState<string>('');

  const handleLogin = async () => {
    try {
      const formData = new URLSearchParams();
      formData.append('username', 'testuser');
      formData.append('password', 'secret');

      const response = await httpClient.post('/token', formData, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
      });

      tokenManager.setToken(response.data.access_token);
      setIsAuthenticated(true);
      setError('');
    } catch (err: any) {
      setError(`Login failed: ${err.response?.data?.detail || err.message}`);
    }
  };

  const handleStartMockRun = async () => {
    try {
      if (!isAuthenticated) {
        setError('Please login first');
        return;
      }

      // This would normally be a real clustering request
      // For testing, we'll create a mock run that simulates progress
      const mockRequest = {
        job_id: 'test-dataset-id',
        methods: ['spectral'],
        n_clusters: 3,
        sigma: 1.0,
        n_neighbors: 10,
        use_pca: false,
        dim_reducer: 'pca',
        random_state: 42
      };

      const response = await httpClient.post('/runs', mockRequest);
      const newRunId = response.data.run_id;
      
      setCurrentRunId(newRunId);
      setError('');
    } catch (err: any) {
      setError(`Failed to start run: ${err.response?.data?.detail || err.message}`);
    }
  };

  const handleTestRunId = () => {
    if (runId.trim()) {
      setCurrentRunId(runId.trim());
      setError('');
    }
  };

  const handleRunComplete = (results: any) => {
    console.log('Run completed:', results);
    setCurrentRunId(null);
  };

  const handleRunError = (errorMsg: string) => {
    console.error('Run error:', errorMsg);
    setError(errorMsg);
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h4" gutterBottom>
        PHASE 2 Test - WebSocket Progress Tracking
      </Typography>

      <Stack spacing={3}>
        {/* Authentication */}
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Authentication
            </Typography>
            {!isAuthenticated ? (
              <Button
                variant="contained"
                onClick={handleLogin}
                startIcon={<PlayArrow />}
              >
                Login as Test User
              </Button>
            ) : (
              <Alert severity="success">
                Authenticated successfully! Token: {tokenManager.getToken()?.substring(0, 20)}...
              </Alert>
            )}
          </CardContent>
        </Card>

        {/* Manual Run ID Testing */}
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Test Specific Run ID
            </Typography>
            <Stack direction="row" spacing={2} alignItems="center">
              <TextField
                label="Run ID"
                value={runId}
                onChange={(e) => setRunId(e.target.value)}
                placeholder="Enter existing run ID"
                size="small"
                sx={{ flexGrow: 1 }}
              />
              <Button
                variant="outlined"
                onClick={handleTestRunId}
                disabled={!runId.trim()}
              >
                Track Progress
              </Button>
            </Stack>
          </CardContent>
        </Card>

        {/* Start New Run */}
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Start New Clustering Run
            </Typography>
            <Button
              variant="contained"
              onClick={handleStartMockRun}
              disabled={!isAuthenticated}
              startIcon={<PlayArrow />}
            >
              Start Mock Clustering Run
            </Button>
          </CardContent>
        </Card>

        {/* Error Display */}
        {error && (
          <Alert severity="error">
            {error}
          </Alert>
        )}

        {/* Progress Tracking */}
        {currentRunId && (
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Live Progress Tracking - Run ID: {currentRunId}
              </Typography>
              <RunProgress
                runId={currentRunId}
                onComplete={handleRunComplete}
                onError={handleRunError}
                showLogs={true}
                autoCollapse={false}
              />
            </CardContent>
          </Card>
        )}

        {/* Instructions */}
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              PHASE 2 Testing Instructions
            </Typography>
            <Typography variant="body2" component="div">
              <ol>
                <li>Click "Login as Test User" to authenticate</li>
                <li>Click "Start Mock Clustering Run" to create a new run</li>
                <li>Watch the real-time progress updates via WebSocket</li>
                <li>If WebSocket fails, it will automatically fallback to polling</li>
                <li>You can also manually enter a run ID to track existing runs</li>
              </ol>
            </Typography>
          </CardContent>
        </Card>
      </Stack>
    </Container>
  );
};

export default Phase2TestPage;
