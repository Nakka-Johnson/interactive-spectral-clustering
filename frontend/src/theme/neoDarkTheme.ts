import { createTheme, alpha } from '@mui/material/styles';
import type { Shadows } from '@mui/material/styles';

const shadows: Shadows = [
  'none',
  '0px 1px 2px rgba(0,0,0,0.30)',
  '0px 1px 3px rgba(0,0,0,0.28)',
  '0px 2px 4px rgba(0,0,0,0.28)',
  '0px 2px 6px rgba(0,0,0,0.30)',
  '0px 3px 8px rgba(0,0,0,0.30)',
  '0px 4px 10px rgba(0,0,0,0.32)',
  '0px 5px 12px rgba(0,0,0,0.34)',
  '0px 6px 14px rgba(0,0,0,0.36)',
  '0px 7px 16px rgba(0,0,0,0.38)',
  '0px 8px 18px rgba(0,0,0,0.40)',
  '0px 9px 20px rgba(0,0,0,0.42)',
  '0px 10px 22px rgba(0,0,0,0.44)',
  '0px 12px 24px rgba(0,0,0,0.46)',
  '0px 14px 28px rgba(0,0,0,0.48)',
  '0px 16px 32px rgba(0,0,0,0.50)',
  '0px 18px 36px rgba(0,0,0,0.50)',
  '0px 20px 40px rgba(0,0,0,0.50)',
  '0px 22px 44px rgba(0,0,0,0.50)',
  '0px 24px 48px rgba(0,0,0,0.50)',
  '0px 28px 56px rgba(0,0,0,0.50)',
  '0px 32px 64px rgba(0,0,0,0.50)',
  '0px 40px 80px rgba(0,0,0,0.50)',
  '0px 48px 96px rgba(0,0,0,0.50)',
  '0px 56px 112px rgba(0,0,0,0.50)',
];

/**
 * Neo Dark Theme for Spectral Analysis Platform
 * Modern dark theme with high contrast and subtle elevations
 */
export const neoDarkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#5B8CFF',
      light: '#7BA3FF',
      dark: '#4A73CC',
      contrastText: '#FFFFFF',
    },
    secondary: {
      main: '#9C6EFF',
      light: '#B088FF',
      dark: '#7C55CC',
      contrastText: '#FFFFFF',
    },
    success: {
      main: '#22C55E',
      light: '#4ADE80',
      dark: '#16A34A',
      contrastText: '#FFFFFF',
    },
    warning: {
      main: '#F59E0B',
      light: '#FBBF24',
      dark: '#D97706',
      contrastText: '#000000',
    },
    error: {
      main: '#EF4444',
      light: '#F87171',
      dark: '#DC2626',
      contrastText: '#FFFFFF',
    },
    background: {
      default: '#0E1116',
      paper: '#151924',
    },
    text: {
      primary: '#F8FAFC',
      secondary: '#CBD5E1',
      disabled: '#64748B',
    },
    divider: alpha('#CBD5E1', 0.12),
    action: {
      active: '#CBD5E1',
      hover: alpha('#F8FAFC', 0.04),
      selected: alpha('#5B8CFF', 0.12),
      disabled: alpha('#F8FAFC', 0.26),
      disabledBackground: alpha('#F8FAFC', 0.12),
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 700,
      fontSize: '2.5rem',
      lineHeight: 1.2,
    },
    h2: {
      fontWeight: 600,
      fontSize: '2rem',
      lineHeight: 1.3,
    },
    h3: {
      fontWeight: 600,
      fontSize: '1.75rem',
      lineHeight: 1.3,
    },
    h4: {
      fontWeight: 600,
      fontSize: '1.5rem',
      lineHeight: 1.4,
    },
    h5: {
      fontWeight: 600,
      fontSize: '1.25rem',
      lineHeight: 1.4,
    },
    h6: {
      fontWeight: 600,
      fontSize: '1.125rem',
      lineHeight: 1.4,
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.6,
    },
    body2: {
      fontSize: '0.875rem',
      lineHeight: 1.6,
    },
    caption: {
      fontSize: '0.75rem',
      lineHeight: 1.5,
    },
  },
  shape: {
    borderRadius: 8,
  },
  shadows,
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          scrollbarColor: '#374151 #111827',
          '&::-webkit-scrollbar, & *::-webkit-scrollbar': {
            backgroundColor: '#111827',
            width: '8px',
          },
          '&::-webkit-scrollbar-thumb, & *::-webkit-scrollbar-thumb': {
            borderRadius: 8,
            backgroundColor: '#374151',
            minHeight: 24,
            border: '1px solid #111827',
          },
          '&::-webkit-scrollbar-thumb:focus, & *::-webkit-scrollbar-thumb:focus': {
            backgroundColor: '#4B5563',
          },
          '&::-webkit-scrollbar-thumb:active, & *::-webkit-scrollbar-thumb:active': {
            backgroundColor: '#4B5563',
          },
          '&::-webkit-scrollbar-thumb:hover, & *::-webkit-scrollbar-thumb:hover': {
            backgroundColor: '#4B5563',
          },
          '&::-webkit-scrollbar-corner, & *::-webkit-scrollbar-corner': {
            backgroundColor: '#111827',
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          borderRadius: 8,
          boxShadow: 'none',
          '&:hover': {
            boxShadow: 'none',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: '#151924',
          borderBottom: '1px solid rgba(203, 213, 225, 0.12)',
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundColor: '#151924',
          borderRight: '1px solid rgba(203, 213, 225, 0.12)',
        },
      },
    },
    MuiListItemButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          margin: '2px 8px',
          '&.Mui-selected': {
            backgroundColor: alpha('#5B8CFF', 0.12),
            color: '#5B8CFF',
            '&:hover': {
              backgroundColor: alpha('#5B8CFF', 0.16),
            },
            '& .MuiListItemIcon-root': {
              color: '#5B8CFF',
            },
          },
          '&:hover': {
            backgroundColor: alpha('#F8FAFC', 0.04),
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 16,
        },
      },
    },
  },
});
