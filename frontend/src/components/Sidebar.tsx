import React from 'react';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Box,
  Typography,
  Divider,
  useTheme,
} from '@mui/material';
import {
  UploadFile,
  Explore,
  Settings,
  Visibility,
  Assessment,
  History,
  PictureAsPdf,
} from '@mui/icons-material';
import { useClusteringStore } from '../store';

const DRAWER_WIDTH = 280;

interface SidebarProps {
  open: boolean;
  onClose: () => void;
}

interface TabItem {
  id: 'upload' | 'explore' | 'configure' | 'visualize' | 'metrics' | 'history' | 'report';
  label: string;
  icon: React.ReactElement;
  disabled?: boolean;
}

const Sidebar: React.FC<SidebarProps> = ({ open, onClose }) => {
  const theme = useTheme();
  const { ui, dataset, currentResult, setCurrentTab } = useClusteringStore();

  const tabs: TabItem[] = [
    {
      id: 'upload',
      label: 'Upload Data',
      icon: <UploadFile />,
    },
    {
      id: 'explore',
      label: 'Data Explore',
      icon: <Explore />,
      disabled: !dataset,
    },
    {
      id: 'configure',
      label: 'Configuration',
      icon: <Settings />,
      disabled: !dataset,
    },
    {
      id: 'visualize',
      label: 'Visualize',
      icon: <Visibility />,
      disabled: !currentResult,
    },
    {
      id: 'metrics',
      label: 'Metrics',
      icon: <Assessment />,
      disabled: !currentResult,
    },
    {
      id: 'history',
      label: 'History',
      icon: <History />,
    },
    {
      id: 'report',
      label: 'Report',
      icon: <PictureAsPdf />,
      disabled: !currentResult,
    },
  ];

  const handleTabClick = (tabId: typeof tabs[0]['id']) => {
    setCurrentTab(tabId);
    onClose();
  };

  const drawerContent = (
    <Box sx={{ width: DRAWER_WIDTH, height: '100%', bgcolor: 'background.paper' }}>
      {/* Header */}
      <Box sx={{ p: 2, bgcolor: 'primary.main', color: 'primary.contrastText' }}>
        <Typography variant="h6" component="div" fontWeight="bold">
          Spectral Clustering
        </Typography>
        <Typography variant="body2" sx={{ opacity: 0.8 }}>
          Interactive Analysis Platform
        </Typography>
      </Box>

      <Divider />

      {/* Navigation Tabs */}
      <List sx={{ pt: 1 }}>
        {tabs.map((tab) => (
          <ListItem key={tab.id} disablePadding>
            <ListItemButton
              selected={ui.currentTab === tab.id}
              disabled={tab.disabled}
              onClick={() => handleTabClick(tab.id)}
              aria-label={`Navigate to ${tab.label} panel${tab.disabled ? ' (disabled)' : ''}`}
              sx={{
                mx: 1,
                my: 0.5,
                borderRadius: 1,
                transition: 'all 0.2s ease-in-out',
                '&:hover': {
                  bgcolor: (theme) => theme.palette.action.hover,
                  transform: 'translateX(4px)',
                  '& .MuiListItemIcon-root': {
                    transform: 'scale(1.1)',
                  },
                },
                '&.Mui-selected': {
                  bgcolor: (theme) => theme.palette.primary.main,
                  color: 'primary.contrastText',
                  boxShadow: (theme) => `0 2px 8px ${theme.palette.primary.main}40`,
                  '&:hover': {
                    bgcolor: (theme) => theme.palette.primary.dark,
                    transform: 'translateX(4px)',
                  },
                },
                '&.Mui-disabled': {
                  opacity: 0.5,
                },
              }}
            >
              <ListItemIcon
                sx={{
                  color: ui.currentTab === tab.id ? 'inherit' : 'action.active',
                  minWidth: 40,
                  transition: 'all 0.2s ease-in-out',
                }}
              >
                {tab.icon}
              </ListItemIcon>
              <ListItemText
                primary={tab.label}
                primaryTypographyProps={{
                  fontSize: '0.9rem',
                  fontWeight: ui.currentTab === tab.id ? 'medium' : 'normal',
                }}
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>

      <Divider sx={{ mt: 2 }} />

      {/* Status Information */}
      <Box sx={{ p: 2, mt: 'auto' }}>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          Status
        </Typography>
        
        {dataset && (
          <Box sx={{ mb: 1 }}>
            <Typography variant="caption" color="text.primary">
              Dataset: {dataset.filename}
            </Typography>
            <br />
            <Typography variant="caption" color="text.secondary">
              {dataset.shape[0]} rows × {dataset.shape[1]} cols
            </Typography>
          </Box>
        )}

        {currentResult && (
          <Box sx={{ mb: 1 }}>
            <Typography variant="caption" color="success.main">
              ✓ Clustering Complete
            </Typography>
          </Box>
        )}

        {!dataset && (
          <Typography variant="caption" color="text.secondary">
            Upload a CSV file to begin
          </Typography>
        )}
      </Box>
    </Box>
  );

  return (
    <>
      {/* Permanent drawer for desktop */}
      <Drawer
        variant="permanent"
        sx={{
          display: { xs: 'none', md: 'block' },
          '& .MuiDrawer-paper': {
            boxSizing: 'border-box',
            width: DRAWER_WIDTH,
            borderRight: `1px solid ${theme.palette.divider}`,
          },
        }}
        open
      >
        {drawerContent}
      </Drawer>

      {/* Temporary drawer for mobile */}
      <Drawer
        variant="temporary"
        open={open}
        onClose={onClose}
        ModalProps={{
          keepMounted: true, // Better mobile performance
        }}
        sx={{
          display: { xs: 'block', md: 'none' },
          '& .MuiDrawer-paper': {
            boxSizing: 'border-box',
            width: DRAWER_WIDTH,
          },
        }}
      >
        {drawerContent}
      </Drawer>
    </>
  );
};

export default Sidebar;
export { DRAWER_WIDTH };
