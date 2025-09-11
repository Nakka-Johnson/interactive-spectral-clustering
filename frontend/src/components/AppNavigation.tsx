/**
 * AppNavigation component for Interactive Spectral Clustering Platform.
 *
 * Provides sidebar navigation with routing links and active state management.
 * Matches the Analysis Platform layout with comprehensive navigation sections.
 */

import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Typography,
  Box,
  Divider,
  Chip,
} from '@mui/material';
import {
  CloudUpload,
  Explore,
  Settings,
  Visibility,
  Analytics,
  Assignment,
} from '@mui/icons-material';

/**
 * Navigation items configuration matching the Analysis Platform layout.
 */
const navigationItems = [
  {
    path: '/upload',
    label: 'Upload',
    icon: CloudUpload,
    description: 'Data Science Toolkit',
  },
  {
    path: '/data-explore',
    label: 'Data Explore',
    icon: Explore,
    description: 'Explore and analyze datasets',
  },
  {
    path: '/config',
    label: 'Config',
    icon: Settings,
    description: 'Configuration settings',
  },
  {
    path: '/visualize',
    label: 'Visualize',
    icon: Visibility,
    description: 'Data visualization',
  },
  {
    path: '/metrics',
    label: 'Metrics',
    icon: Analytics,
    description: 'Performance metrics',
  },
  {
    path: '/report',
    label: 'Report',
    icon: Assignment,
    description: 'Generate reports',
  },
];

/**
 * AppNavigation component with Analysis Platform layout.
 */
export const AppNavigation: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();

  const handleNavigation = (path: string) => {
    navigate(path);
  };

  const isActivePath = (path: string) => location.pathname === path;

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: 200,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: 200,
          boxSizing: 'border-box',
          backgroundImage: 'linear-gradient(180deg, #111111 0%, #1a1a1a 100%)',
          borderRight: '1px solid #333333',
          color: '#ffffff',
        },
      }}
    >
      {/* Header */}
      <Box sx={{ p: 2, pb: 1 }}>
        <Typography
          variant="h6"
          component="h1"
          sx={{
            fontWeight: 600,
            color: '#00e5ff',
            fontSize: '1.1rem',
            background: 'linear-gradient(45deg, #00e5ff 30%, #ff6ec7 90%)',
            backgroundClip: 'text',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}
        >
          Analysis Platform
        </Typography>
        <Typography
          variant="caption"
          sx={{
            color: '#b3b3b3',
            fontSize: '0.75rem',
          }}
        >
          Data Science Toolkit
        </Typography>
      </Box>

      {/* Status Info */}
      <Box sx={{ px: 2, pb: 1 }}>
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <Chip
            label="Connected"
            size="small"
            sx={{
              bgcolor: '#00e5ff',
              color: '#000000',
              fontSize: '0.7rem',
              height: 20,
              fontWeight: 600,
              boxShadow: '0 0 10px rgba(0, 229, 255, 0.3)',
            }}
          />
          <Typography
            variant="caption"
            sx={{
              color: '#b3b3b3',
              fontSize: '0.7rem',
            }}
          >
            Dataset: 890000 samples
          </Typography>
        </Box>
        <Typography
          variant="caption"
          sx={{
            color: '#b3b3b3',
            fontSize: '0.7rem',
          }}
        >
          Features: 17 | Analyses: 3
        </Typography>
      </Box>

      <Divider sx={{ mx: 1, borderColor: '#333333' }} />

      {/* Navigation Menu */}
      <List sx={{ flex: 1, py: 1 }}>
        {navigationItems.map((item) => {
          const IconComponent = item.icon;
          const isActive = isActivePath(item.path);

          return (
            <ListItem key={item.path} disablePadding>
              <ListItemButton
                onClick={() => handleNavigation(item.path)}
                sx={{
                  mx: 1,
                  borderRadius: 1,
                  minHeight: 40,
                  bgcolor: isActive ? 'rgba(0, 229, 255, 0.1)' : 'transparent',
                  borderLeft: isActive ? '3px solid #00e5ff' : '3px solid transparent',
                  '&:hover': {
                    bgcolor: 'rgba(0, 229, 255, 0.05)',
                    borderLeft: '3px solid rgba(0, 229, 255, 0.3)',
                  },
                  pl: 1.5,
                }}
                selected={isActive}
              >
                <ListItemIcon
                  sx={{
                    color: isActive ? '#00e5ff' : '#b3b3b3',
                    minWidth: 32,
                    '& .MuiSvgIcon-root': {
                      fontSize: '1.2rem',
                    },
                  }}
                >
                  <IconComponent />
                </ListItemIcon>
                <ListItemText
                  primary={item.label}
                  primaryTypographyProps={{
                    fontWeight: isActive ? 600 : 400,
                    fontSize: '0.875rem',
                    color: isActive ? '#00e5ff' : '#ffffff',
                  }}
                />
              </ListItemButton>
            </ListItem>
          );
        })}
      </List>

      {/* Footer Status */}
      <Box sx={{ p: 2, mt: 'auto' }}>
        <Divider sx={{ borderColor: '#333333', mb: 1 }} />
        <Typography
          variant="caption"
          sx={{
            color: '#b3b3b3',
            fontSize: '0.7rem',
          }}
        >
          Ready for analysis
        </Typography>
        <br />
        <Typography
          variant="caption"
          sx={{
            color: '#b3b3b3',
            fontSize: '0.7rem',
          }}
        >
          API: localhost:8001 ●
        </Typography>
      </Box>
    </Drawer>
  );
};

export default AppNavigation;
