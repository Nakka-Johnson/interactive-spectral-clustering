/**
 * Simple FileUpload component for Interactive Spectral Clustering Platform.
 */

import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Box,
  Button,
  Card,
  CardContent,
  Typography,
  TextField,
  LinearProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Stack
} from '@mui/material';
import { CloudUpload } from '@mui/icons-material';
import { useAppStore, useIsUploading, useUploadProgress } from '../../store/appStore';

interface FilePreview {
  name: string;
  size: number;
  data: string[][];
  headers: string[];
  totalRows: number;
}

interface FileUploadProps {
  onUploadSuccess?: (jobId: string) => void;
  onUploadError?: (error: string) => void;
  maxPreviewRows?: number;
}

export const FileUpload: React.FC<FileUploadProps> = ({
  onUploadSuccess,
  onUploadError,
  maxPreviewRows = 10
}) => {
  const { uploadDataset, setError, clearError } = useAppStore();
  const isUploading = useIsUploading();
  const uploadProgress = useUploadProgress();
  
  const [filePreview, setFilePreview] = useState<FilePreview | null>(null);
  const [datasetName, setDatasetName] = useState('');
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'success' | 'error'>('idle');

  const parseCSV = useCallback((content: string): { data: string[][]; headers: string[] } => {
    const lines = content.split('\n').filter(line => line.trim());
    if (lines.length === 0) {
      throw new globalThis.Error('File is empty');
    }
    
    const rows = lines.map(line => line.split(',').map(cell => cell.trim()));
    const headers = rows[0];
    const data = rows.slice(1);
    
    return { headers, data };
  }, []);

  const handleFileAccepted = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;
    
    const file = acceptedFiles[0];
    clearError();
    setUploadStatus('idle');
    
    if (!datasetName) {
      setDatasetName(file.name.replace(/\.[^/.]+$/, ''));
    }
    
    try {
      const content = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target?.result as string);
        reader.onerror = () => reject(new globalThis.Error('Failed to read file'));
        reader.readAsText(file);
      });
      
      let parsedData: { data: string[][]; headers: string[] };
      
      if (file.type === 'text/csv' || file.name.endsWith('.csv')) {
        parsedData = parseCSV(content);
      } else {
        throw new globalThis.Error('Unsupported file type. Please upload CSV files.');
      }
      
      const previewData = parsedData.data.slice(0, maxPreviewRows);
      
      setFilePreview({
        name: file.name,
        size: file.size,
        data: previewData,
        headers: parsedData.headers,
        totalRows: parsedData.data.length
      });
      
    } catch (error: any) {
      const errorMessage = error?.message || 'Failed to preview file';
      setError(errorMessage);
      onUploadError?.(errorMessage);
    }
  }, [parseCSV, datasetName, maxPreviewRows, clearError, setError, onUploadError]);

  const { getRootProps, getInputProps, isDragActive, fileRejections } = useDropzone({
    onDrop: handleFileAccepted,
    accept: {
      'text/csv': ['.csv']
    },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024,
    disabled: isUploading
  });

  const handleUpload = useCallback(async () => {
    if (!filePreview || isUploading) return;
    
    try {
      const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
      const file = fileInput?.files?.[0];
      
      if (!file) {
        throw new globalThis.Error('No file selected');
      }
      
      const response = await uploadDataset(file, datasetName || filePreview.name);
      setUploadStatus('success');
      onUploadSuccess?.(response.job_id);
      
      setTimeout(() => {
        setFilePreview(null);
        setDatasetName('');
        setUploadStatus('idle');
      }, 2000);
      
    } catch (error: any) {
      const errorMessage = error?.message || 'Upload failed';
      setUploadStatus('error');
      onUploadError?.(errorMessage);
    }
  }, [filePreview, isUploading, uploadDataset, datasetName, onUploadSuccess, onUploadError]);

  const handleClear = useCallback(() => {
    setFilePreview(null);
    setDatasetName('');
    setUploadStatus('idle');
    clearError();
  }, [clearError]);

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <Box sx={{ maxWidth: 800, mx: 'auto', p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Upload Dataset
      </Typography>
      
      <Typography variant="body1" color="text.secondary" paragraph>
        Upload a CSV file to start clustering analysis. 
        The file should contain numeric data with column headers.
      </Typography>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box
            {...getRootProps()}
            sx={{
              border: 2,
              borderStyle: 'dashed',
              borderColor: isDragActive ? 'primary.main' : 'grey.300',
              borderRadius: 2,
              p: 4,
              textAlign: 'center',
              cursor: isUploading ? 'not-allowed' : 'pointer',
              bgcolor: isDragActive ? 'action.hover' : 'background.paper',
              transition: 'all 0.2s ease',
              '&:hover': {
                borderColor: 'primary.main',
                bgcolor: 'action.hover'
              }
            }}
          >
            <input {...getInputProps()} />
            
            <CloudUpload sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
            
            {isDragActive ? (
              <Typography variant="h6" color="primary">
                Drop the file here...
              </Typography>
            ) : (
              <>
                <Typography variant="h6" gutterBottom>
                  Drag & drop a CSV file here, or click to select
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Supports CSV files up to 10MB
                </Typography>
              </>
            )}
          </Box>

          {fileRejections.length > 0 && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {fileRejections[0].errors[0].message}
            </Alert>
          )}
        </CardContent>
      </Card>

      {filePreview && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              {filePreview.name}
            </Typography>

            <Stack direction="row" spacing={1} sx={{ mb: 2 }}>
              <Chip label={`${filePreview.totalRows} rows`} size="small" />
              <Chip label={`${filePreview.headers.length} columns`} size="small" />
              <Chip label={formatFileSize(filePreview.size)} size="small" />
            </Stack>

            <TextField
              fullWidth
              label="Dataset Name"
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
              sx={{ mb: 2 }}
              disabled={isUploading}
              helperText="Give your dataset a meaningful name"
            />

            {isUploading && (
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Uploading... {uploadProgress}%
                </Typography>
                <LinearProgress variant="determinate" value={uploadProgress} />
              </Box>
            )}

            <Button
              variant="contained"
              onClick={handleUpload}
              disabled={isUploading || !datasetName.trim()}
              startIcon={<CloudUpload />}
              sx={{ mb: 2, mr: 2 }}
            >
              {isUploading ? 'Uploading...' : 'Upload Dataset'}
            </Button>

            <Button
              variant="outlined"
              onClick={handleClear}
              disabled={isUploading}
            >
              Clear
            </Button>

            <Typography variant="subtitle1" gutterBottom sx={{ mt: 2 }}>
              Preview ({Math.min(maxPreviewRows, filePreview.totalRows)} of {filePreview.totalRows} rows)
            </Typography>
            
            <TableContainer component={Paper} sx={{ maxHeight: 400 }}>
              <Table stickyHeader size="small">
                <TableHead>
                  <TableRow>
                    {filePreview.headers.map((header, index) => (
                      <TableCell key={index} sx={{ fontWeight: 'bold' }}>
                        {header}
                      </TableCell>
                    ))}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {filePreview.data.map((row, rowIndex) => (
                    <TableRow key={rowIndex}>
                      {row.map((cell, cellIndex) => (
                        <TableCell key={cellIndex}>
                          {cell}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      )}

      {uploadStatus === 'success' && (
        <Alert severity="success" sx={{ mb: 2 }}>
          Dataset uploaded successfully! You can now use it for clustering analysis.
        </Alert>
      )}
      
      {uploadStatus === 'error' && (
        <Alert severity="error" sx={{ mb: 2 }}>
          Upload failed. Please try again or check your file format.
        </Alert>
      )}
    </Box>
  );
};

export default FileUpload;
