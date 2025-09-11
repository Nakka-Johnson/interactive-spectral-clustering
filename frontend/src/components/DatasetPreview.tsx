/**
 * DatasetPreview component for Interactive Spectral Clustering Platform.
 * Shows server-side preview of dataset with pagination and data grid.
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  TableContainer,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  TablePagination,
  Paper,
  Chip,
  Stack,
  IconButton,
  Tooltip,
  Alert,
  CircularProgress,
} from '@mui/material';
import { Visibility, PlayArrow, Info, TableChart, Refresh } from '@mui/icons-material';
import { getDatasetPreview } from '../api';

interface DatasetPreviewProps {
  datasetId: string;
  onLoadForAnalysis?: (datasetId: string) => void;
}

interface PreviewData {
  columns: string[];
  rows: any[][];
  totalRows: number;
  totalColumns: number;
  previewRows: number;
}

export const DatasetPreview: React.FC<DatasetPreviewProps> = ({ datasetId, onLoadForAnalysis }) => {
  const [previewData, setPreviewData] = useState<PreviewData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);

  useEffect(() => {
    loadPreview();
  }, [datasetId]);

  const loadPreview = async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await getDatasetPreview(datasetId);
      setPreviewData(data);
    } catch (err: any) {
      setError(err.response?.data?.message || err.message || 'Failed to load preview');
    } finally {
      setLoading(false);
    }
  };

  const handleChangePage = (event: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const handleLoadForAnalysis = () => {
    onLoadForAnalysis?.(datasetId);
  };

  const paginatedRows =
    previewData?.rows.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage) || [];

  if (loading) {
    return (
      <Card>
        <CardContent>
          <Box display="flex" alignItems="center" justifyContent="center" py={4}>
            <Stack spacing={2} alignItems="center">
              <CircularProgress />
              <Typography variant="body1" color="text.secondary">
                Loading dataset preview...
              </Typography>
            </Stack>
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent>
          <Alert
            severity="error"
            action={
              <Button color="inherit" size="small" onClick={loadPreview} startIcon={<Refresh />}>
                Retry
              </Button>
            }
          >
            <Typography variant="body1">Failed to load dataset preview</Typography>
            <Typography variant="caption" display="block">
              {error}
            </Typography>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  if (!previewData) {
    return null;
  }

  return (
    <Card>
      <CardContent>
        <Stack spacing={3}>
          {/* Header */}
          <Box display="flex" alignItems="center" justifyContent="space-between">
            <Box display="flex" alignItems="center" gap={2}>
              <TableChart color="primary" fontSize="large" />
              <Box>
                <Typography variant="h6">Dataset Preview</Typography>
                <Typography variant="caption" color="text.secondary">
                  Showing first {previewData.previewRows} rows of {previewData.totalRows} total
                </Typography>
              </Box>
            </Box>

            <Button
              variant="contained"
              startIcon={<PlayArrow />}
              onClick={handleLoadForAnalysis}
              size="large"
            >
              Load Data for Analysis
            </Button>
          </Box>

          {/* Dataset Info */}
          <Box display="flex" gap={2} flexWrap="wrap">
            <Chip
              icon={<TableChart />}
              label={`${previewData.totalColumns} columns`}
              color="primary"
              variant="outlined"
            />
            <Chip
              icon={<Visibility />}
              label={`${previewData.totalRows} rows`}
              color="secondary"
              variant="outlined"
            />
            <Tooltip title="Server-side preview limited to first 50 rows">
              <Chip
                icon={<Info />}
                label={`Preview: ${previewData.previewRows} rows`}
                color="default"
                variant="outlined"
              />
            </Tooltip>
          </Box>

          {/* Data Table */}
          <Paper variant="outlined">
            <TableContainer sx={{ maxHeight: 400 }}>
              <Table stickyHeader size="small">
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ minWidth: 60, fontWeight: 'bold' }}>#</TableCell>
                    {previewData.columns.map((column, index) => (
                      <TableCell key={index} sx={{ minWidth: 120, fontWeight: 'bold' }}>
                        {column}
                      </TableCell>
                    ))}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {paginatedRows.map((row, rowIndex) => (
                    <TableRow
                      key={page * rowsPerPage + rowIndex}
                      hover
                      sx={{ '&:nth-of-type(odd)': { backgroundColor: 'action.hover' } }}
                    >
                      <TableCell sx={{ color: 'text.secondary', fontFamily: 'monospace' }}>
                        {page * rowsPerPage + rowIndex + 1}
                      </TableCell>
                      {row.map((cell, cellIndex) => (
                        <TableCell key={cellIndex}>
                          <Typography variant="body2" noWrap>
                            {cell !== null && cell !== undefined ? String(cell) : '—'}
                          </Typography>
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>

            <TablePagination
              component="div"
              count={previewData.previewRows}
              page={page}
              onPageChange={handleChangePage}
              rowsPerPage={rowsPerPage}
              onRowsPerPageChange={handleChangeRowsPerPage}
              rowsPerPageOptions={[5, 10, 25, 50]}
              labelRowsPerPage="Rows per page:"
              showFirstButton
              showLastButton
            />
          </Paper>

          {/* Note */}
          <Alert severity="info" icon={<Info />}>
            <Typography variant="body2">
              This is a preview of your dataset. Click "Load Data for Analysis" to proceed to
              algorithm configuration.
            </Typography>
          </Alert>
        </Stack>
      </CardContent>
    </Card>
  );
};

export default DatasetPreview;
