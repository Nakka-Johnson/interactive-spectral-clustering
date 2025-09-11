/**
 * PageHeader component for consistent page headers across the platform.
 */

import React from 'react';
import { Box, Typography, Chip } from '@mui/material';

interface PageHeaderProps {
  title: string;
  subtitle?: string;
  status?: 'connected' | 'disconnected' | 'loading';
  stats?: {
    samples?: number;
    features?: number;
    analyses?: number;
  };
}

export const PageHeader: React.FC<PageHeaderProps> = ({
  title,
  subtitle,
  status = 'connected',
  stats,
}) => {
  const getStatusColor = () => {
    switch (status) {
      case 'connected':
        return '#00e5ff';
      case 'disconnected':
        return '#ff6ec7';
      case 'loading':
        return '#ffab00';
      default:
        return '#757575';
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'connected':
        return 'Connected';
      case 'disconnected':
        return 'Disconnected';
      case 'loading':
        return 'Connecting...';
      default:
        return 'Unknown';
    }
  };

  return (
    <Box
      sx={{
        background: 'linear-gradient(135deg, #1a1a1a 0%, #262626 100%)',
        borderBottom: '1px solid #333333',
        px: 3,
        py: 2,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        backdropFilter: 'blur(10px)',
      }}
    >
      <Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 0.5 }}>
          <Typography
            variant="h5"
            component="h1"
            sx={{
              fontWeight: 600,
              background: 'linear-gradient(45deg, #00e5ff 30%, #ff6ec7 90%)',
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              fontSize: '1.5rem',
            }}
          >
            {title}
          </Typography>
          <Chip
            label={getStatusText()}
            size="small"
            sx={{
              bgcolor: getStatusColor(),
              color: status === 'connected' ? '#000000' : '#ffffff',
              fontSize: '0.75rem',
              height: 24,
              fontWeight: 600,
              boxShadow: `0 0 10px ${getStatusColor()}33`,
            }}
          />
        </Box>
        {subtitle && (
          <Typography
            variant="body2"
            sx={{
              color: '#b3b3b3',
              fontSize: '0.875rem',
            }}
          >
            {subtitle}
          </Typography>
        )}
      </Box>

      {stats && (
        <Box sx={{ textAlign: 'right' }}>
          <Typography variant="caption" sx={{ color: '#b3b3b3', fontSize: '0.75rem' }}>
            {stats.samples && `Dataset: ${stats.samples.toLocaleString()} samples`}
          </Typography>
          <br />
          <Typography variant="caption" sx={{ color: '#b3b3b3', fontSize: '0.75rem' }}>
            {stats.features && `Features: ${stats.features}`}
            {stats.analyses && ` | Analyses: ${stats.analyses}`}
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default PageHeader;
