import React from 'react';
import {
  Box,
  LinearProgress,
  Typography,
  Card,
  CardContent,
  Alert,
  Chip,
  Button,
  CircularProgress,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
} from '@mui/material';
import {
  CheckCircle,
  Cancel,
  Schedule,
  PlayArrow,
  Stop,
  Info,
  Warning,
  Error as ErrorIcon,
} from '@mui/icons-material';
import { useClusteringStore } from '../store';

interface ProgressStep {
  id: string;
  label: string;
  description?: string;
  status: 'pending' | 'running' | 'completed' | 'error';
  progress?: number;
  message?: string;
}

const ProgressBar: React.FC = () => {
  const {
    progress: progressState,
    setProgress,
  } = useClusteringStore();

  const { isRunning, progress } = progressState;

  // Mock logs and current step - in real app these would be in the store
  const logs = [
    { level: 'info', message: 'Starting clustering analysis...', timestamp: Date.now() },
    { level: 'info', message: 'Preprocessing data...', timestamp: Date.now() },
    { level: 'warning', message: 'High memory usage detected', timestamp: Date.now() },
  ];

  const getCurrentStep = () => {
    if (progress < 20) return 'preprocessing';
    if (progress < 40) return 'dimensionality_reduction';
    if (progress < 70) return 'clustering';
    if (progress < 90) return 'evaluation';
    return 'visualization';
  };

  const currentStep = getCurrentStep();

  const stopClustering = () => {
    setProgress({ isRunning: false, message: 'Analysis stopped by user' });
  };

  // Mock progress steps - in a real app this would come from the store
  const steps: ProgressStep[] = [
    {
      id: 'preprocessing',
      label: 'Data Preprocessing',
      description: 'Cleaning and normalizing data',
      status: currentStep === 'preprocessing' ? 'running' : 
              progress > 0 ? 'completed' : 'pending',
      progress: currentStep === 'preprocessing' ? progress : 
                progress > 0 ? 100 : 0,
    },
    {
      id: 'dimensionality_reduction',
      label: 'Dimensionality Reduction',
      description: 'Applying PCA/t-SNE/UMAP',
      status: currentStep === 'dimensionality_reduction' ? 'running' :
              progress > 25 ? 'completed' : 'pending',
      progress: currentStep === 'dimensionality_reduction' ? progress :
                progress > 25 ? 100 : 0,
    },
    {
      id: 'clustering',
      label: 'Clustering Analysis',
      description: 'Running selected algorithms',
      status: currentStep === 'clustering' ? 'running' :
              progress > 50 ? 'completed' : 'pending',
      progress: currentStep === 'clustering' ? progress :
                progress > 50 ? 100 : 0,
    },
    {
      id: 'evaluation',
      label: 'Evaluation Metrics',
      description: 'Computing clustering quality metrics',
      status: currentStep === 'evaluation' ? 'running' :
              progress > 75 ? 'completed' : 'pending',
      progress: currentStep === 'evaluation' ? progress :
                progress > 75 ? 100 : 0,
    },
    {
      id: 'visualization',
      label: 'Generating Visualizations',
      description: 'Creating plots and reports',
      status: currentStep === 'visualization' ? 'running' :
              progress >= 100 ? 'completed' : 'pending',
      progress: currentStep === 'visualization' ? progress :
                progress >= 100 ? 100 : 0,
    },
  ];

  const getStatusIcon = (status: ProgressStep['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircle color="success" />;
      case 'running':
        return <CircularProgress size={24} />;
      case 'error':
        return <Cancel color="error" />;
      default:
        return <Schedule color="disabled" />;
    }
  };

  if (!isRunning && progress === 0) {
    return (
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="h6" color="text.secondary">
          Configure parameters and click "Run Clustering Analysis" to start
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Analysis Progress
      </Typography>
      
      {/* Overall Progress */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
            <Typography variant="h6">
              Overall Progress
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Chip
                label={isRunning ? 'Running' : progress >= 100 ? 'Completed' : 'Stopped'}
                color={isRunning ? 'primary' : progress >= 100 ? 'success' : 'default'}
                icon={isRunning ? <PlayArrow /> : progress >= 100 ? <CheckCircle /> : <Stop />}
              />
              {isRunning && (
                <Button
                  variant="outlined"
                  color="error"
                  startIcon={<Stop />}
                  onClick={stopClustering}
                  size="small"
                >
                  Stop Analysis
                </Button>
              )}
            </Box>
          </Box>

          <Box sx={{ mb: 2 }}>
            <LinearProgress
              variant="determinate"
              value={progress}
              sx={{ height: 8, borderRadius: 4 }}
            />
          </Box>

          <Typography variant="body2" color="text.secondary" align="center">
            {progress.toFixed(1)}% Complete
          </Typography>
        </CardContent>
      </Card>

      {/* Detailed Steps */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Detailed Progress
          </Typography>

          <Stepper orientation="vertical">
            {steps.map((step, index) => (
              <Step key={step.id} active={step.status === 'running'} completed={step.status === 'completed'}>
                <StepLabel
                  icon={getStatusIcon(step.status)}
                  error={step.status === 'error'}
                >
                  <Typography variant="subtitle1">
                    {step.label}
                  </Typography>
                </StepLabel>
                <StepContent>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                    {step.description}
                  </Typography>
                  
                  {step.status === 'running' && (
                    <Box sx={{ mb: 1 }}>
                      <LinearProgress
                        variant="indeterminate"
                        sx={{ height: 4, borderRadius: 2 }}
                      />
                    </Box>
                  )}
                  
                  {step.message && (
                    <Alert
                      severity={step.status === 'error' ? 'error' : 'info'}
                      sx={{ mt: 1 }}
                    >
                      {step.message}
                    </Alert>
                  )}
                </StepContent>
              </Step>
            ))}
          </Stepper>
        </CardContent>
      </Card>

      {/* Live Logs */}
      {logs && logs.length > 0 && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Live Logs
            </Typography>
            
            <Box
              sx={{
                maxHeight: 300,
                overflow: 'auto',
                bgcolor: 'grey.50',
                p: 2,
                borderRadius: 1,
                fontFamily: 'monospace',
              }}
            >
              <List dense>
                {logs.slice(-10).map((log, index) => (
                  <React.Fragment key={index}>
                    <ListItem sx={{ py: 0.5 }}>
                      <ListItemIcon sx={{ minWidth: 36 }}>
                        {log.level === 'error' ? (
                          <ErrorIcon color="error" fontSize="small" />
                        ) : log.level === 'warning' ? (
                          <Warning color="warning" fontSize="small" />
                        ) : (
                          <Info color="info" fontSize="small" />
                        )}
                      </ListItemIcon>
                      <ListItemText
                        primary={log.message}
                        secondary={new Date(log.timestamp).toLocaleTimeString()}
                        primaryTypographyProps={{
                          variant: 'body2',
                          fontFamily: 'monospace',
                        }}
                        secondaryTypographyProps={{
                          variant: 'caption',
                        }}
                      />
                    </ListItem>
                    {index < logs.slice(-10).length - 1 && <Divider />}
                  </React.Fragment>
                ))}
              </List>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Resource Usage */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Resource Usage
          </Typography>
          
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <Box>
              <Typography variant="body2" gutterBottom>
                CPU Usage
              </Typography>
              <LinearProgress
                variant="determinate"
                value={Math.random() * 100} // Mock data
                color="primary"
                sx={{ height: 6, borderRadius: 3 }}
              />
            </Box>
            
            <Box>
              <Typography variant="body2" gutterBottom>
                Memory Usage
              </Typography>
              <LinearProgress
                variant="determinate"
                value={Math.random() * 100} // Mock data
                color="secondary"
                sx={{ height: 6, borderRadius: 3 }}
              />
            </Box>
            
            <Box>
              <Typography variant="body2" gutterBottom>
                GPU Usage
              </Typography>
              <LinearProgress
                variant="determinate"
                value={Math.random() * 100} // Mock data
                color="success"
                sx={{ height: 6, borderRadius: 3 }}
              />
            </Box>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default ProgressBar;
