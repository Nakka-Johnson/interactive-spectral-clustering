import React from 'react';
import { Box, Typography } from '@mui/material';

const VisualizationPage: React.FC = () => {
  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Clustering Visualization
      </Typography>
      <Typography variant="body1" color="text.secondary">
        Interactive visualization of clustering results.
      </Typography>
      <Typography variant="body2" sx={{ mt: 2 }}>
        This page will contain the visualization interface.
      </Typography>
    </Box>
  );
};

export default VisualizationPage;
