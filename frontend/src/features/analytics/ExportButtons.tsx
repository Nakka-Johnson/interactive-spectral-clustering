/**
 * ExportButtons.tsx
 * 
 * Export buttons component for clustering run results.
 * Provides download options for labels CSV, HTML reports, and ZIP bundles.
 */

import React, { useState } from 'react';
import {
  Button,
  ButtonGroup,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  CircularProgress,
  Alert,
  Snackbar
} from '@mui/material';
import {
  Download as DownloadIcon,
  TableChart as TableChartIcon,
  Description as DescriptionIcon,
  Archive as ArchiveIcon,
  GetApp as GetAppIcon
} from '@mui/icons-material';

interface ExportButtonsProps {
  runId: string;
  disabled?: boolean;
  variant?: 'text' | 'outlined' | 'contained';
  size?: 'small' | 'medium' | 'large';
}

interface ExportOption {
  key: string;
  label: string;
  icon: React.ReactNode;
  endpoint: string;
  filename: string;
  description: string;
}

const ExportButtons: React.FC<ExportButtonsProps> = ({
  runId,
  disabled = false,
  variant = 'outlined',
  size = 'medium'
}) => {
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const exportOptions: ExportOption[] = [
    {
      key: 'labels',
      label: 'Labels CSV',
      icon: <TableChartIcon />,
      endpoint: `/api/runs/${runId}/labels.csv`,
      filename: `labels_${runId}.csv`,
      description: 'Download clustering labels with original data as CSV'
    },
    {
      key: 'report',
      label: 'HTML Report',
      icon: <DescriptionIcon />,
      endpoint: `/api/runs/${runId}/report`,
      filename: `report_${runId}.html`,
      description: 'Download comprehensive analysis report (print-ready)'
    },
    {
      key: 'bundle',
      label: 'Complete Bundle',
      icon: <ArchiveIcon />,
      endpoint: `/api/runs/${runId}/bundle.zip`,
      filename: `bundle_${runId}.zip`,
      description: 'Download complete analysis bundle (CSV + HTML + JSON)'
    }
  ];

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const downloadFile = async (url: string, filename: string) => {
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(url, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (!response.ok) {
        throw new Error(`Download failed: ${response.statusText}`);
      }

      // Get the blob data
      const blob = await response.blob();
      
      // Create download link
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = filename;
      
      // Trigger download
      document.body.appendChild(link);
      link.click();
      
      // Cleanup
      document.body.removeChild(link);
      window.URL.revokeObjectURL(downloadUrl);
      
      return true;
    } catch (err) {
      console.error('Download error:', err);
      throw err;
    }
  };

  const handleExport = async (option: ExportOption) => {
    try {
      setLoading(option.key);
      setError(null);
      handleMenuClose();

      await downloadFile(option.endpoint, option.filename);
      
      setSuccess(`${option.label} downloaded successfully`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Download failed');
    } finally {
      setLoading(null);
    }
  };

  const handleCloseSnackbar = () => {
    setError(null);
    setSuccess(null);
  };

  return (
    <>
      <ButtonGroup variant={variant} size={size} disabled={disabled}>
        <Button
          onClick={handleMenuOpen}
          startIcon={loading ? <CircularProgress size={16} /> : <DownloadIcon />}
          disabled={disabled || loading !== null}
        >
          Export
        </Button>
        <Button
          size="small"
          onClick={handleMenuOpen}
          disabled={disabled || loading !== null}
          sx={{ px: 1 }}
        >
          <GetAppIcon fontSize="small" />
        </Button>
      </ButtonGroup>

      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'left',
        }}
        transformOrigin={{
          vertical: 'top',
          horizontal: 'left',
        }}
      >
        {exportOptions.map((option) => (
          <MenuItem
            key={option.key}
            onClick={() => handleExport(option)}
            disabled={loading === option.key}
          >
            <ListItemIcon>
              {loading === option.key ? (
                <CircularProgress size={20} />
              ) : (
                option.icon
              )}
            </ListItemIcon>
            <ListItemText
              primary={option.label}
              secondary={option.description}
            />
          </MenuItem>
        ))}
      </Menu>

      {/* Success/Error Snackbars */}
      <Snackbar
        open={Boolean(success)}
        autoHideDuration={4000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert severity="success" onClose={handleCloseSnackbar}>
          {success}
        </Alert>
      </Snackbar>

      <Snackbar
        open={Boolean(error)}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert severity="error" onClose={handleCloseSnackbar}>
          {error}
        </Alert>
      </Snackbar>
    </>
  );
};

export default ExportButtons;
