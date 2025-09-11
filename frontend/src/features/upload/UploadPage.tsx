/**
 * UploadPage component for Interactive Spectral Clustering Platform.
 * Combines file upload and dataset preview with summary statistics.
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Card,
  CardContent,
  Button,
  Alert,
  Paper,
  Stack,
  Chip,
  Divider,
} from '@mui/material';
import Grid from '@mui/material/Grid';
import {
  CloudUpload,
  CheckCircle,
  Storage,
  ViewColumn,
  TableRows,
  FolderOpen,
  Memory,
} from '@mui/icons-material';
import FileUploader from '../../components/FileUploader';
import DatasetPreview from '../../components/DatasetPreview';

interface DatasetStats {
  totalRows: number;
  totalColumns: number;
  selectedFeatures: number;
  fileSizeKB: number;
}

export const UploadPage: React.FC = () => {
  const [uploadedDatasetId, setUploadedDatasetId] = useState<string | null>(null);
  const [datasetStats, setDatasetStats] = useState<DatasetStats | null>(null);
  const [showSuccessToast, setShowSuccessToast] = useState(false);

  const handleUploadSuccess = (datasetId: string, stats: any) => {
    setUploadedDatasetId(datasetId);
    setDatasetStats({
      totalRows: stats.total_rows || 0,
      totalColumns: stats.total_columns || 0,
      selectedFeatures: stats.total_columns || 0, // Initially all features selected
      fileSizeKB: stats.file_size_kb || 0,
    });
    setShowSuccessToast(true);

    // Hide success toast after 5 seconds
    setTimeout(() => setShowSuccessToast(false), 5000);
  };

  const handleUploadError = (error: string) => {
    setUploadedDatasetId(null);
    setDatasetStats(null);
    setShowSuccessToast(false);
  };

  const handleLoadForAnalysis = (datasetId: string) => {
    // Navigate to configuration page
    console.log('Loading dataset for analysis:', datasetId);
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Stack spacing={4}>
        {/* Page Header */}
        <Box>
          <Typography variant="h4" gutterBottom>
            Data Upload & Processing
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Upload your dataset and preview the data before running clustering analysis.
          </Typography>
        </Box>

        {/* Data Upload & Processing Card */}
        <Card elevation={2}>
          <CardContent>
            <Box display="flex" alignItems="center" justifyContent="space-between" mb={3}>
              <Box display="flex" alignItems="center" gap={2}>
                <Storage color="primary" fontSize="large" />
                <Box>
                  <Typography variant="h6">Data Upload & Processing</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Secure file upload and automated data validation
                  </Typography>
                </Box>
              </Box>

              <Chip icon={<CheckCircle />} label="Connected" color="success" variant="outlined" />
            </Box>

            <FileUploader onUploadSuccess={handleUploadSuccess} onUploadError={handleUploadError} />
          </CardContent>
        </Card>

        {/* Success Toast */}
        {showSuccessToast && (
          <Alert
            severity="success"
            onClose={() => setShowSuccessToast(false)}
            icon={<CheckCircle />}
          >
            <Typography variant="body1">File successfully uploaded to backend</Typography>
            <Typography variant="caption" display="block">
              Dataset is ready for preview and analysis
            </Typography>
          </Alert>
        )}

        {/* Dataset Preview */}
        {uploadedDatasetId && (
          <DatasetPreview datasetId={uploadedDatasetId} onLoadForAnalysis={handleLoadForAnalysis} />
        )}

        {/* Summary Cards */}
        {datasetStats && (
          <Box>
            <Typography variant="h6" gutterBottom>
              Dataset Summary
            </Typography>
            <Grid container spacing={3}>
              <Grid item xs={12} sm={6} md={3}>
                <Paper sx={{ p: 3, textAlign: 'center' }}>
                  <Stack spacing={2} alignItems="center">
                    <TableRows color="primary" fontSize="large" />
                    <Box>
                      <Typography variant="h4" color="primary.main">
                        {datasetStats.totalRows.toLocaleString()}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Total Rows
                      </Typography>
                    </Box>
                  </Stack>
                </Paper>
              </Grid>

              <Grid item xs={12} sm={6} md={3}>
                <Paper sx={{ p: 3, textAlign: 'center' }}>
                  <Stack spacing={2} alignItems="center">
                    <ViewColumn color="secondary" fontSize="large" />
                    <Box>
                      <Typography variant="h4" color="secondary.main">
                        {datasetStats.totalColumns}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Total Columns
                      </Typography>
                    </Box>
                  </Stack>
                </Paper>
              </Grid>

              <Grid item xs={12} sm={6} md={3}>
                <Paper sx={{ p: 3, textAlign: 'center' }}>
                  <Stack spacing={2} alignItems="center">
                    <FolderOpen color="info" fontSize="large" />
                    <Box>
                      <Typography variant="h4" color="info.main">
                        {datasetStats.selectedFeatures}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Selected Features
                      </Typography>
                    </Box>
                  </Stack>
                </Paper>
              </Grid>

              <Grid item xs={12} sm={6} md={3}>
                <Paper sx={{ p: 3, textAlign: 'center' }}>
                  <Stack spacing={2} alignItems="center">
                    <Memory color="warning" fontSize="large" />
                    <Box>
                      <Typography variant="h4" color="warning.main">
                        {datasetStats.fileSizeKB.toFixed(1)}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        File Size (KB)
                      </Typography>
                    </Box>
                  </Stack>
                </Paper>
              </Grid>
            </Grid>
          </Box>
        )}

        {/* Help Section */}
        {!uploadedDatasetId && (
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Getting Started
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Upload a CSV file to begin the clustering analysis process. Your data will be
                processed server-side and you'll get an immediate preview before proceeding to
                algorithm configuration.
              </Typography>
              <Divider sx={{ my: 2 }} />
              <Typography variant="subtitle2" gutterBottom>
                Supported File Formats:
              </Typography>
              <Stack direction="row" spacing={1}>
                <Chip label="CSV" size="small" variant="outlined" />
                <Chip label="UTF-8 Encoding" size="small" variant="outlined" />
                <Chip label="Header Row Required" size="small" variant="outlined" />
              </Stack>
            </CardContent>
          </Card>
        )}
      </Stack>
    </Container>
  );
};

export default UploadPage;
