import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Paper,
  Grid,
  Chip,
  Alert,
  Snackbar,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  LinearProgress,
  Divider,
  IconButton,
} from '@mui/material';
import {
  PictureAsPdf as PdfIcon,
  Download as DownloadIcon,
  Assessment as AssessmentIcon,
  Analytics as AnalyticsIcon,
  Engineering as EngineeringIcon,
  Compare as CompareIcon,
  Description as DescriptionIcon,
  Dataset as DatasetIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';

interface ReportPreview {
  report_type: string;
  sections: string[];
  data_summary: {
    dataset_records: number;
    analyses_count: number;
    estimated_pages: number;
  };
}

interface ReportType {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  color: 'primary' | 'secondary' | 'info' | 'warning';
}

const REPORT_TYPES: ReportType[] = [
  {
    id: 'executive',
    name: 'Executive Summary',
    description: 'High-level overview with key findings and recommendations for stakeholders',
    icon: <AssessmentIcon />,
    color: 'primary',
  },
  {
    id: 'detailed',
    name: 'Detailed Analysis',
    description: 'Comprehensive analysis including methodology, results, and performance metrics',
    icon: <AnalyticsIcon />,
    color: 'secondary',
  },
  {
    id: 'technical',
    name: 'Technical Report',
    description: 'In-depth technical documentation with implementation details and code',
    icon: <EngineeringIcon />,
    color: 'info',
  },
  {
    id: 'comparison',
    name: 'Algorithm Comparison',
    description: 'Comparative analysis of different clustering algorithms and their performance',
    icon: <CompareIcon />,
    color: 'warning',
  },
];

export const ReportPage: React.FC = () => {
  const [selectedReportType, setSelectedReportType] = useState<string>('executive');
  const [experimentName, setExperimentName] = useState('Clustering Analysis Report');
  const [isGenerating, setIsGenerating] = useState(false);
  const [reportPreview, setReportPreview] = useState<ReportPreview | null>(null);
  const [snackbar, setSnackbar] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error' | 'info' | 'warning';
  }>({
    open: false,
    message: '',
    severity: 'success',
  });

  // API functions
  const fetchReportPreview = async (reportType: string): Promise<ReportPreview> => {
    const response = await fetch(`http://localhost:8000/export/report/preview?report_type=${reportType}`);
    if (!response.ok) {
      throw new Error('Failed to fetch report preview');
    }
    return response.json();
  };

  const generateReport = async (reportType: string, experimentName: string): Promise<void> => {
    const response = await fetch('http://localhost:8000/export/report', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        report_type: reportType,
        experiment_name: experimentName,
        include_sections: ['executive_summary', 'dataset_overview', 'methods', 'results', 'metrics', 'conclusions'],
      }),
    });

    if (!response.ok) {
      throw new Error('Failed to generate report');
    }

    // Download the PDF
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${reportType}_report_${Date.now()}.pdf`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  };

  // Load preview when report type changes
  useEffect(() => {
    fetchReportPreview(selectedReportType)
      .then(setReportPreview)
      .catch((error) => {
        console.error('Error fetching report preview:', error);
        setSnackbar({
          open: true,
          message: 'Failed to load report preview',
          severity: 'error',
        });
      });
  }, [selectedReportType]);

  const handleGenerateReport = async () => {
    try {
      setIsGenerating(true);
      await generateReport(selectedReportType, experimentName);
      
      setSnackbar({
        open: true,
        message: 'PDF report generated successfully!',
        severity: 'success',
      });
    } catch (error) {
      console.error('Error generating report:', error);
      setSnackbar({
        open: true,
        message: 'Failed to generate report. Please try again.',
        severity: 'error',
      });
    } finally {
      setIsGenerating(false);
    }
  };

  const handleRefreshPreview = () => {
    fetchReportPreview(selectedReportType)
      .then(setReportPreview)
      .catch((error) => {
        console.error('Error refreshing preview:', error);
        setSnackbar({
          open: true,
          message: 'Failed to refresh preview',
          severity: 'error',
        });
      });
  };

  const selectedReportTypeData = REPORT_TYPES.find(type => type.id === selectedReportType);

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Report Generation
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Generate comprehensive PDF reports with selectable content and formatting options
      </Typography>

      <Grid container spacing={3}>
        {/* Left Panel - Configuration */}
        <Grid item xs={12} md={8}>
          <Grid container spacing={3}>
            {/* Report Type Selection */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Select Report Type
                  </Typography>
                  
                  <Grid container spacing={2}>
                    {REPORT_TYPES.map((reportType) => (
                      <Grid item xs={12} sm={6} key={reportType.id}>
                        <Paper
                          sx={{
                            p: 2,
                            cursor: 'pointer',
                            border: selectedReportType === reportType.id ? 2 : 1,
                            borderColor: selectedReportType === reportType.id 
                              ? `${reportType.color}.main` 
                              : 'grey.300',
                            '&:hover': {
                              borderColor: `${reportType.color}.main`,
                              boxShadow: 2,
                            },
                          }}
                          onClick={() => setSelectedReportType(reportType.id)}
                        >
                          <Box display="flex" alignItems="center" mb={1}>
                            <Box sx={{ color: `${reportType.color}.main`, mr: 1 }}>
                              {reportType.icon}
                            </Box>
                            <Typography variant="subtitle1" fontWeight="medium">
                              {reportType.name}
                            </Typography>
                          </Box>
                          <Typography variant="body2" color="text.secondary">
                            {reportType.description}
                          </Typography>
                        </Paper>
                      </Grid>
                    ))}
                  </Grid>
                </CardContent>
              </Card>
            </Grid>

            {/* Report Configuration */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Report Configuration
                  </Typography>
                  
                  <Grid container spacing={2}>
                    <Grid item xs={12}>
                      <TextField
                        label="Experiment Name"
                        value={experimentName}
                        onChange={(e) => setExperimentName(e.target.value)}
                        fullWidth
                        placeholder="Enter a descriptive name for your analysis"
                      />
                    </Grid>
                    
                    <Grid item xs={12}>
                      <FormControl fullWidth>
                        <InputLabel>Report Format</InputLabel>
                        <Select
                          value="pdf"
                          label="Report Format"
                          disabled
                        >
                          <MenuItem value="pdf">PDF Document</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>

            {/* Current Selection Summary */}
            {selectedReportTypeData && (
              <Grid item xs={12}>
                <Card variant="outlined">
                  <CardContent>
                    <Box display="flex" alignItems="center" mb={2}>
                      <Box sx={{ color: `${selectedReportTypeData.color}.main`, mr: 1 }}>
                        {selectedReportTypeData.icon}
                      </Box>
                      <Typography variant="h6">
                        {selectedReportTypeData.name}
                      </Typography>
                    </Box>
                    <Typography variant="body2" color="text.secondary" mb={2}>
                      {selectedReportTypeData.description}
                    </Typography>
                    
                    <Box display="flex" gap={1}>
                      <Chip 
                        label={`${selectedReportTypeData.name}`} 
                        color={selectedReportTypeData.color} 
                        size="small" 
                      />
                      <Chip 
                        label="PDF Format" 
                        variant="outlined" 
                        size="small" 
                      />
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            )}

            {/* Generate Button */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Box display="flex" justifyContent="space-between" alignItems="center">
                    <Box>
                      <Typography variant="h6" gutterBottom>
                        Generate Report
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Create and download your PDF report
                      </Typography>
                    </Box>
                    
                    <Button
                      variant="contained"
                      size="large"
                      startIcon={isGenerating ? null : <PdfIcon />}
                      onClick={handleGenerateReport}
                      disabled={isGenerating || !experimentName.trim()}
                      sx={{ minWidth: 200 }}
                    >
                      {isGenerating ? (
                        <>
                          <LinearProgress sx={{ width: 100, mr: 1 }} />
                          Generating...
                        </>
                      ) : (
                        'Generate PDF Report'
                      )}
                    </Button>
                  </Box>
                  
                  {isGenerating && (
                    <Box sx={{ mt: 2 }}>
                      <Alert severity="info">
                        Generating your PDF report. This may take a few moments...
                      </Alert>
                    </Box>
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Grid>

        {/* Right Panel - Preview */}
        <Grid item xs={12} md={4}>
          <Card sx={{ height: 'fit-content', position: 'sticky', top: 20 }}>
            <CardContent>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                <Typography variant="h6">
                  Report Preview
                </Typography>
                <IconButton onClick={handleRefreshPreview} size="small">
                  <RefreshIcon />
                </IconButton>
              </Box>
              
              {/* Included Sections */}
              {reportPreview && (
                <>
                  <Typography variant="subtitle2" gutterBottom>
                    Included Sections
                  </Typography>
                  <List dense sx={{ mb: 2 }}>
                    {reportPreview.sections.map((section, index) => (
                      <ListItem key={index} sx={{ py: 0.5 }}>
                        <ListItemIcon sx={{ minWidth: 32 }}>
                          <CheckCircleIcon color="success" fontSize="small" />
                        </ListItemIcon>
                        <ListItemText 
                          primary={section} 
                          primaryTypographyProps={{ variant: 'body2' }}
                        />
                      </ListItem>
                    ))}
                  </List>
                  
                  <Divider sx={{ my: 2 }} />
                  
                  {/* Data Summary */}
                  <Typography variant="subtitle2" gutterBottom>
                    Data Summary
                  </Typography>
                  <Box sx={{ mb: 2 }}>
                    <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                      <Box display="flex" alignItems="center">
                        <DatasetIcon fontSize="small" sx={{ mr: 1, color: 'text.secondary' }} />
                        <Typography variant="body2">Dataset Records</Typography>
                      </Box>
                      <Typography variant="body2" fontWeight="medium">
                        {reportPreview.data_summary.dataset_records.toLocaleString()}
                      </Typography>
                    </Box>
                    
                    <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                      <Box display="flex" alignItems="center">
                        <AnalyticsIcon fontSize="small" sx={{ mr: 1, color: 'text.secondary' }} />
                        <Typography variant="body2">Analyses Count</Typography>
                      </Box>
                      <Typography variant="body2" fontWeight="medium">
                        {reportPreview.data_summary.analyses_count}
                      </Typography>
                    </Box>
                    
                    <Box display="flex" justifyContent="space-between" alignItems="center">
                      <Box display="flex" alignItems="center">
                        <DescriptionIcon fontSize="small" sx={{ mr: 1, color: 'text.secondary' }} />
                        <Typography variant="body2">Estimated Pages</Typography>
                      </Box>
                      <Typography variant="body2" fontWeight="medium">
                        {reportPreview.data_summary.estimated_pages}
                      </Typography>
                    </Box>
                  </Box>
                  
                  <Alert severity="info">
                    Preview shows estimated content based on current data
                  </Alert>
                </>
              )}
              
              {!reportPreview && (
                <Box textAlign="center" py={3}>
                  <LinearProgress sx={{ mb: 2 }} />
                  <Typography variant="body2" color="text.secondary">
                    Loading preview...
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Success/Error Snackbar */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
      >
        <Alert 
          onClose={() => setSnackbar({ ...snackbar, open: false })} 
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default ReportPage;
