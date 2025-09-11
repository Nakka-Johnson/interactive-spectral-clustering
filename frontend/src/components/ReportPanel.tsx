import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  FormControl,
  Select,
  MenuItem,
  InputLabel,
  Checkbox,
  FormControlLabel,
  FormGroup,
  TextField,
  Alert,
  Stepper,
  Step,
  StepLabel,
  LinearProgress,
  Divider,
} from '@mui/material';
import {
  PictureAsPdf,
  Download,
  Email,
  Print,
  Share,
  Settings,
  Assessment,
  Image,
  TableChart,
  Analytics,
} from '@mui/icons-material';
import { useClusteringStore } from '../store';

interface ReportConfig {
  format: 'pdf' | 'html' | 'docx';
  sections: {
    summary: boolean;
    parameters: boolean;
    visualizations: boolean;
    metrics: boolean;
    rawData: boolean;
    methodology: boolean;
  };
  includeCode: boolean;
  includeTimestamps: boolean;
  template: 'technical' | 'executive' | 'research';
  customTitle: string;
  customDescription: string;
}

const ReportPanel: React.FC = () => {
  const { currentResult, parameters, dataset, experiments } = useClusteringStore();
  
  const [config, setConfig] = useState<ReportConfig>({
    format: 'pdf',
    sections: {
      summary: true,
      parameters: true,
      visualizations: true,
      metrics: true,
      rawData: false,
      methodology: true,
    },
    includeCode: false,
    includeTimestamps: true,
    template: 'technical',
    customTitle: 'Clustering Analysis Report',
    customDescription: 'Comprehensive analysis of clustering results',
  });

  const [isGenerating, setIsGenerating] = useState(false);
  const [generationProgress, setGenerationProgress] = useState(0);
  const [activeStep, setActiveStep] = useState(0);

  const handleConfigChange = (key: keyof ReportConfig, value: any) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  };

  const handleSectionChange = (section: keyof ReportConfig['sections'], checked: boolean) => {
    setConfig(prev => ({
      ...prev,
      sections: { ...prev.sections, [section]: checked }
    }));
  };

  const generateReport = async () => {
    setIsGenerating(true);
    setGenerationProgress(0);
    setActiveStep(0);

    const steps = [
      'Collecting data',
      'Generating visualizations',
      'Computing statistics',
      'Formatting document',
      'Finalizing report'
    ];

    // Simulate report generation
    for (let i = 0; i < steps.length; i++) {
      setActiveStep(i);
      await new Promise(resolve => setTimeout(resolve, 1000));
      setGenerationProgress((i + 1) * 20);
    }

    // In real app, this would generate and download the actual report
    console.log('Report generated with config:', config);
    
    setIsGenerating(false);
    setActiveStep(0);
    setGenerationProgress(0);
  };

  const previewReport = () => {
    console.log('Preview report with config:', config);
  };

  const emailReport = () => {
    console.log('Email report');
  };

  const printReport = () => {
    console.log('Print report');
  };

  const shareReport = () => {
    console.log('Share report');
  };

  const getSectionIcon = (section: string) => {
    const icons: Record<string, React.ReactNode> = {
      summary: <Assessment />,
      parameters: <Settings />,
      visualizations: <Image />,
      metrics: <Analytics />,
      rawData: <TableChart />,
      methodology: <Assessment />,
    };
    return icons[section] || <Assessment />;
  };

  const getEstimatedSize = () => {
    let size = 2; // Base size in MB
    if (config.sections.visualizations) size += 5;
    if (config.sections.rawData) size += 10;
    if (config.sections.metrics) size += 1;
    if (config.includeCode) size += 2;
    return size.toFixed(1);
  };

  const getSectionCount = () => {
    return Object.values(config.sections).filter(Boolean).length;
  };

  if (!currentResult && experiments.length === 0) {
    return (
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="h6" color="text.secondary">
          Run clustering analysis to generate reports
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Report Generation
      </Typography>
      
      {/* Report Configuration */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Report Configuration
          </Typography>
          
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            {/* Basic Settings */}
            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
              <FormControl sx={{ minWidth: 150 }}>
                <InputLabel>Format</InputLabel>
                <Select
                  value={config.format}
                  onChange={(e) => handleConfigChange('format', e.target.value)}
                  aria-label="Select report output format"
                  tabIndex={0}
                >
                  <MenuItem value="pdf">PDF</MenuItem>
                  <MenuItem value="html">HTML</MenuItem>
                  <MenuItem value="docx">Word Document</MenuItem>
                </Select>
              </FormControl>

              <FormControl sx={{ minWidth: 150 }}>
                <InputLabel>Template</InputLabel>
                <Select
                  value={config.template}
                  onChange={(e) => handleConfigChange('template', e.target.value)}
                  aria-label="Select report template style"
                  tabIndex={0}
                >
                  <MenuItem value="technical">Technical</MenuItem>
                  <MenuItem value="executive">Executive Summary</MenuItem>
                  <MenuItem value="research">Research Paper</MenuItem>
                </Select>
              </FormControl>
            </Box>

            {/* Custom Title and Description */}
            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
              <TextField
                label="Report Title"
                value={config.customTitle}
                onChange={(e) => handleConfigChange('customTitle', e.target.value)}
                aria-label="Enter custom report title"
                inputProps={{ tabIndex: 0 }}
                sx={{ flex: 1, minWidth: 200 }}
              />
              <TextField
                label="Description"
                value={config.customDescription}
                onChange={(e) => handleConfigChange('customDescription', e.target.value)}
                aria-label="Enter report description"
                inputProps={{ tabIndex: 0 }}
                sx={{ flex: 1, minWidth: 200 }}
                multiline
                rows={2}
              />
            </Box>
          </Box>
        </CardContent>
      </Card>

      {/* Section Selection */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Report Sections
          </Typography>
          
          <FormGroup>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              {Object.entries(config.sections).map(([section, checked]) => (
                <FormControlLabel
                  key={section}
                  control={
                    <Checkbox
                      checked={checked}
                      onChange={(e) => handleSectionChange(section as keyof ReportConfig['sections'], e.target.checked)}
                      aria-label={`Toggle ${section} section in report`}
                      tabIndex={0}
                    />
                  }
                  label={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      {getSectionIcon(section)}
                      <Typography>
                        {section.charAt(0).toUpperCase() + section.slice(1).replace(/([A-Z])/g, ' $1')}
                      </Typography>
                    </Box>
                  }
                />
              ))}
            </Box>
          </FormGroup>

          <Divider sx={{ my: 2 }} />

          {/* Additional Options */}
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            <FormControlLabel
              control={
                <Checkbox
                  checked={config.includeCode}
                  onChange={(e) => handleConfigChange('includeCode', e.target.checked)}
                />
              }
              label="Include code snippets and technical details"
            />
            <FormControlLabel
              control={
                <Checkbox
                  checked={config.includeTimestamps}
                  onChange={(e) => handleConfigChange('includeTimestamps', e.target.checked)}
                />
              }
              label="Include timestamps and execution metadata"
            />
          </Box>
        </CardContent>
      </Card>

      {/* Report Summary */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Report Summary
          </Typography>
          
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
            <Typography variant="body2" color="text.secondary">
              <strong>{getSectionCount()}</strong> sections selected
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Estimated size: <strong>{getEstimatedSize()} MB</strong>
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Format: <strong>{config.format.toUpperCase()}</strong>
            </Typography>
          </Box>

          {dataset && (
            <Alert severity="info" sx={{ mb: 2 }}>
              Report will include analysis of <strong>{dataset.filename}</strong> with{' '}
              <strong>{parameters.methods.length}</strong> clustering algorithm(s).
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Generation Progress */}
      {isGenerating && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Generating Report...
            </Typography>
            
            <Box sx={{ mb: 2 }}>
              <LinearProgress
                variant="determinate"
                value={generationProgress}
                sx={{ height: 8, borderRadius: 4 }}
              />
              <Typography variant="body2" color="text.secondary" align="center" sx={{ mt: 1 }}>
                {generationProgress}% Complete
              </Typography>
            </Box>

            <Stepper activeStep={activeStep} orientation="vertical">
              {[
                'Collecting data',
                'Generating visualizations',
                'Computing statistics',
                'Formatting document',
                'Finalizing report'
              ].map((label) => (
                <Step key={label}>
                  <StepLabel>{label}</StepLabel>
                </Step>
              ))}
            </Stepper>
          </CardContent>
        </Card>
      )}

      {/* Action Buttons */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Generate Report
          </Typography>
          
          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
            <Button
              variant="contained"
              startIcon={<PictureAsPdf />}
              onClick={generateReport}
              disabled={isGenerating || getSectionCount() === 0}
              size="large"
              aria-label="Generate report with selected sections and format"
              tabIndex={0}
            >
              Generate Report
            </Button>
            
            <Button
              variant="outlined"
              startIcon={<Download />}
              onClick={previewReport}
              disabled={isGenerating}
              aria-label="Preview report before generating"
              tabIndex={0}
            >
              Preview
            </Button>
            
            <Button
              variant="outlined"
              startIcon={<Email />}
              onClick={emailReport}
              disabled={isGenerating}
              aria-label="Email report to recipients"
              tabIndex={0}
            >
              Email
            </Button>
            
            <Button
              variant="outlined"
              startIcon={<Print />}
              onClick={printReport}
              disabled={isGenerating}
              aria-label="Print report"
              tabIndex={0}
            >
              Print
            </Button>
            
            <Button
              variant="outlined"
              startIcon={<Share />}
              onClick={shareReport}
              disabled={isGenerating}
              aria-label="Share report link"
              tabIndex={0}
            >
              Share
            </Button>
          </Box>

          {getSectionCount() === 0 && (
            <Alert severity="warning" sx={{ mt: 2 }}>
              Please select at least one section to include in the report.
            </Alert>
          )}
        </CardContent>
      </Card>

      {!currentResult && (
        <Alert severity="info" sx={{ mt: 2 }}>
          Generate your first clustering analysis to create comprehensive reports.
        </Alert>
      )}
    </Box>
  );
};

export default ReportPanel;
