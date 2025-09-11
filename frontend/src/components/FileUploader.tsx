/**
 * FileUploader component for Interactive Spectral Clustering Platform.
 * Provides drag-and-drop and click-to-select CSV file upload functionality.
 */

import React, { useCallback, useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Alert,
  LinearProgress,
  Stack,
  IconButton,
} from '@mui/material';
import { CloudUpload, AttachFile, CheckCircle, Error as ErrorIcon } from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import { uploadDataset } from '../api';

interface FileUploaderProps {
  onUploadSuccess?: (datasetId: string, stats: any) => void;
  onUploadError?: (error: string) => void;
}

export const FileUploader: React.FC<FileUploaderProps> = ({ onUploadSuccess, onUploadError }) => {
  const [uploading, setUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (!file) return;

      setUploadedFile(file);
      setUploading(true);
      setUploadError(null);
      setUploadSuccess(false);

      try {
        const response = await uploadDataset(file);
        setUploadSuccess(true);
        setUploading(false);
        onUploadSuccess?.(response.dataset_id, response.stats);
      } catch (error: any) {
        const errorMessage = error.response?.data?.message || error.message || 'Upload failed';
        setUploadError(errorMessage);
        setUploading(false);
        onUploadError?.(errorMessage);
      }
    },
    [onUploadSuccess, onUploadError],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.ms-excel': ['.csv'],
    },
    multiple: false,
    disabled: uploading,
  });

  const handleReset = () => {
    setUploadSuccess(false);
    setUploadError(null);
    setUploadedFile(null);
  };

  return (
    <Card>
      <CardContent>
        <Stack spacing={3}>
          <Box display="flex" alignItems="center" gap={2}>
            <CloudUpload color="primary" fontSize="large" />
            <Typography variant="h6">Upload Dataset</Typography>
          </Box>

          {!uploadSuccess && !uploading && (
            <Card
              variant="outlined"
              sx={{
                border: '2px dashed',
                borderColor: isDragActive ? 'primary.main' : 'divider',
                backgroundColor: isDragActive ? 'action.hover' : 'transparent',
                cursor: 'pointer',
                transition: 'all 0.2s ease-in-out',
                '&:hover': {
                  borderColor: 'primary.main',
                  backgroundColor: 'action.hover',
                },
              }}
            >
              <CardContent
                {...getRootProps()}
                sx={{
                  textAlign: 'center',
                  py: 4,
                }}
              >
                <input {...getInputProps()} />
                <Stack spacing={2} alignItems="center">
                  <AttachFile sx={{ fontSize: 48, color: 'text.secondary' }} />

                  <Typography variant="h6" color="text.primary">
                    {isDragActive ? 'Drop the CSV file here' : 'Drag & drop a CSV file here'}
                  </Typography>

                  <Typography variant="body2" color="text.secondary">
                    or
                  </Typography>

                  <Button variant="outlined" component="span" startIcon={<CloudUpload />}>
                    Click to select file
                  </Button>

                  <Typography variant="caption" color="text.secondary">
                    Supported formats: CSV files only
                  </Typography>
                </Stack>
              </CardContent>
            </Card>
          )}

          {uploading && (
            <Box>
              <Stack spacing={2}>
                <Box display="flex" alignItems="center" gap={2}>
                  <Typography variant="body1">Uploading {uploadedFile?.name}...</Typography>
                </Box>
                <LinearProgress />
                <Typography variant="caption" color="text.secondary">
                  Processing file and extracting dataset information
                </Typography>
              </Stack>
            </Box>
          )}

          {uploadSuccess && (
            <Alert
              severity="success"
              icon={<CheckCircle />}
              action={
                <Button color="inherit" size="small" onClick={handleReset}>
                  Upload Another
                </Button>
              }
            >
              <Typography variant="body1">File successfully uploaded to backend</Typography>
              {uploadedFile && (
                <Typography variant="caption" display="block">
                  {uploadedFile.name} ({(uploadedFile.size / 1024).toFixed(1)} KB)
                </Typography>
              )}
            </Alert>
          )}

          {uploadError && (
            <Alert
              severity="error"
              icon={<ErrorIcon />}
              action={
                <Button color="inherit" size="small" onClick={handleReset}>
                  Try Again
                </Button>
              }
            >
              <Typography variant="body1">Upload failed</Typography>
              <Typography variant="caption" display="block">
                {uploadError}
              </Typography>
            </Alert>
          )}
        </Stack>
      </CardContent>
    </Card>
  );
};

export default FileUploader;
