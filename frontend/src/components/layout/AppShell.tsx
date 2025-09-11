import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Box,
  AppBar,
  Toolbar,
  Typography,
  Drawer,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Chip,
  IconButton,
  useMediaQuery,
  useTheme,
} from '@mui/material';
import {
  Menu as MenuIcon,
  CloudUpload,
  DataObject,
  Settings,
  Visibility,
  Analytics,
  Storage,
  Business,
  History,
  Description,
} from '@mui/icons-material';

const drawerWidth = 280;

interface NavigationItem {
  id: string;
  label: string;
  icon: React.ReactElement;
  path: string;
}

const navigationItems: NavigationItem[] = [
  {
    id: 'upload',
    label: 'Upload',
    icon: <CloudUpload />,
    path: '/upload',
  },
  {
    id: 'data-explore',
    label: 'Data Explore',
    icon: <DataObject />,
    path: '/data-explore',
  },
  {
    id: 'config',
    label: 'Config',
    icon: <Settings />,
    path: '/config',
  },
  {
    id: 'visualize',
    label: 'Visualize',
    icon: <Visibility />,
    path: '/visualize',
  },
  {
    id: 'reports',
    label: 'Reports',
    icon: <Description />,
    path: '/reports',
  },
  {
    id: 'metrics',
    label: 'Metrics',
    icon: <Analytics />,
    path: '/metrics',
  },
  {
    id: 'big-data',
    label: 'Big Data',
    icon: <Storage />,
    path: '/big-data',
  },
  {
    id: 'enterprise',
    label: 'Enterprise',
    icon: <Business />,
    path: '/enterprise',
  },
  {
    id: 'history',
    label: 'History',
    icon: <History />,
    path: '/history',
  },
  {
    id: 'report',
    label: 'Report',
    icon: <Description />,
    path: '/report',
  },
];

interface AppShellProps {
  children: React.ReactNode;
}

export const AppShell: React.FC<AppShellProps> = ({ children }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const location = useLocation();
  const navigate = useNavigate();
  const [mobileOpen, setMobileOpen] = useState(false);

  // Get the current active item based on the current route
  const getActiveItem = () => {
    const currentPath = location.pathname;
    const activeItem = navigationItems.find(item => item.path === currentPath);
    return activeItem?.id || 'upload';
  };

  const [selectedItem, setSelectedItem] = useState(getActiveItem());

  // Update selected item when route changes
  useEffect(() => {
    setSelectedItem(getActiveItem());
  }, [location.pathname]);

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleNavItemClick = (item: NavigationItem) => {
    setSelectedItem(item.id);
    navigate(item.path);
    if (isMobile) {
      setMobileOpen(false);
    }
  };

  const drawer = (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Navigation Items */}
      <Box sx={{ flex: 1, pt: 2 }}>
        <List>
          {navigationItems.map((item) => (
            <ListItemButton
              key={item.id}
              selected={selectedItem === item.id}
              onClick={() => handleNavItemClick(item)}
              sx={{
                minHeight: 48,
                px: 2.5,
                mx: 1,
                mb: 0.5,
                borderRadius: 2,
              }}
            >
              <ListItemIcon
                sx={{
                  minWidth: 0,
                  mr: 3,
                  justifyContent: 'center',
                  color: selectedItem === item.id ? 'primary.main' : 'text.secondary',
                }}
              >
                {item.icon}
              </ListItemIcon>
              <ListItemText
                primary={item.label}
                sx={{
                  '& .MuiListItemText-primary': {
                    fontSize: '0.875rem',
                    fontWeight: selectedItem === item.id ? 600 : 400,
                    color: selectedItem === item.id ? 'primary.main' : 'text.primary',
                  },
                }}
              />
            </ListItemButton>
          ))}
        </List>
      </Box>
    </Box>
  );

  return (
    <Box sx={{ display: 'flex' }}>
      {/* App Bar */}
      <AppBar
        position="fixed"
        sx={{
          width: { md: `calc(100% - ${drawerWidth}px)` },
          ml: { md: `${drawerWidth}px` },
          zIndex: theme.zIndex.drawer + 1,
        }}
      >
        <Toolbar>
          {isMobile && (
            <IconButton
              color="inherit"
              aria-label="open drawer"
              edge="start"
              onClick={handleDrawerToggle}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>
          )}
          <Typography
            variant="h6"
            noWrap
            component="div"
            sx={{
              flexGrow: 1,
              fontWeight: 600,
              fontSize: '1.125rem',
            }}
          >
            Spectral Analysis Platform
          </Typography>
          <Chip
            label="Connected"
            color="success"
            size="small"
            sx={{
              fontWeight: 500,
              fontSize: '0.75rem',
            }}
          />
        </Toolbar>
      </AppBar>

      {/* Navigation Drawer */}
      <Box
        component="nav"
        sx={{ width: { md: drawerWidth }, flexShrink: { md: 0 } }}
        aria-label="navigation menu"
      >
        {/* Mobile drawer */}
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true, // Better open performance on mobile.
          }}
          sx={{
            display: { xs: 'block', md: 'none' },
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: drawerWidth,
            },
          }}
        >
          {drawer}
        </Drawer>

        {/* Desktop drawer */}
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', md: 'block' },
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: drawerWidth,
            },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>

      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          width: { md: `calc(100% - ${drawerWidth}px)` },
          minHeight: '100vh',
          bgcolor: 'background.default',
        }}
      >
        <Toolbar /> {/* Spacer for fixed AppBar */}
        <Box sx={{ p: 3 }}>{children}</Box>
      </Box>
    </Box>
  );
};
