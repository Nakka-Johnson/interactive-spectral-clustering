import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Chip,
  Button,
  IconButton,
  Tooltip,
  TextField,
  InputAdornment,
  Menu,
  MenuItem,
  Checkbox,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Divider,
} from '@mui/material';
import { DataGrid, GridColDef, GridRenderCellParams } from '@mui/x-data-grid';
import {
  History,
  Search,
  FilterList,
  MoreVert,
  Delete,
  Edit,
  Visibility,
  Compare,
  Download,
  Star,
  CalendarToday,
  Speed,
} from '@mui/icons-material';
import { useClusteringStore } from '../store';

const HistoryPanel: React.FC = () => {
  const { experiments, removeExperiment, loadExperiment, setUI } = useClusteringStore();
  
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedExperiments, setSelectedExperiments] = useState<string[]>([]);
  const [actionMenu, setActionMenu] = useState<{ element: HTMLElement; experimentId: string } | null>(null);
  const [deleteDialog, setDeleteDialog] = useState<string | null>(null);
  const [editDialog, setEditDialog] = useState<string | null>(null);
  const [editNotes, setEditNotes] = useState('');

  // Mock experiments data if none exist
  const mockExperiments = experiments.length === 0 ? [
    {
      id: '1',
      timestamp: new Date('2024-01-15T10:30:00'),
      jobId: 'job_001',
      parameters: {
        methods: ['kmeans', 'spectral'],
        n_clusters: 3,
        sigma: 1.0,
        n_neighbors: 10,
        use_pca: false,
        dim_reducer: 'pca' as const,
      },
      filters: { useAllColumns: true },
      result: {
        labels: { kmeans: [0, 1, 2, 0, 1], spectral: [1, 0, 2, 1, 0] },
        coords2D: [[1, 2], [3, 4], [5, 6]],
        coords3D: [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        metrics: { kmeans: { silhouette: 0.65 }, spectral: { silhouette: 0.71 } },
      },
      metrics: { kmeans: { silhouette: 0.65 }, spectral: { silhouette: 0.71 } },
      executionTime: 5.2,
      notes: 'Initial clustering experiment with default parameters',
    },
    {
      id: '2',
      timestamp: new Date('2024-01-14T14:20:00'),
      jobId: 'job_002',
      parameters: {
        methods: ['dbscan'],
        n_clusters: 4,
        sigma: 1.5,
        n_neighbors: 15,
        use_pca: true,
        dim_reducer: 'tsne' as const,
      },
      filters: { useAllColumns: true },
      result: {
        labels: { dbscan: [0, 1, -1, 0, 2] },
        coords2D: [[2, 3], [4, 5], [6, 7]],
        coords3D: [[2, 3, 4], [5, 6, 7], [8, 9, 10]],
        metrics: { dbscan: { silhouette: 0.58 } },
      },
      metrics: { dbscan: { silhouette: 0.58 } },
      executionTime: 3.8,
      notes: 'DBSCAN with PCA preprocessing',
    },
  ] : experiments;

  const filteredExperiments = mockExperiments.filter(exp =>
    exp.notes?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    exp.parameters.methods.some(method => method.toLowerCase().includes(searchTerm.toLowerCase())) ||
    exp.jobId.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleSelectExperiment = (experimentId: string, checked: boolean) => {
    if (checked) {
      setSelectedExperiments(prev => [...prev, experimentId]);
    } else {
      setSelectedExperiments(prev => prev.filter(id => id !== experimentId));
    }
  };

  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      setSelectedExperiments(filteredExperiments.map(exp => exp.id));
    } else {
      setSelectedExperiments([]);
    }
  };

  const handleLoadExperiment = (experimentId: string) => {
    loadExperiment(experimentId);
    setUI({ currentTab: 'results' });
  };

  const handleDeleteExperiment = (experimentId: string) => {
    removeExperiment(experimentId);
    setDeleteDialog(null);
    setSelectedExperiments(prev => prev.filter(id => id !== experimentId));
  };

  const handleBulkDelete = () => {
    selectedExperiments.forEach(id => removeExperiment(id));
    setSelectedExperiments([]);
  };

  const handleCompareSelected = () => {
    setUI({ selectedExperiments, comparisonMode: true, currentTab: 'results' });
  };

  const handleEditNotes = (experimentId: string) => {
    const experiment = mockExperiments.find(exp => exp.id === experimentId);
    if (experiment) {
      setEditNotes(experiment.notes || '');
      setEditDialog(experimentId);
    }
  };

  const formatDate = (date: Date) => {
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getBestMetric = (metrics: Record<string, Record<string, number>>) => {
    let bestScore = -1;
    let bestMethod = '';
    
    Object.entries(metrics).forEach(([method, scores]) => {
      const silhouette = scores.silhouette || 0;
      if (silhouette > bestScore) {
        bestScore = silhouette;
        bestMethod = method;
      }
    });
    
    return { method: bestMethod, score: bestScore };
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Experiment History
      </Typography>
      
      {/* Search and Controls */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', mb: 2 }}>
            <TextField
              placeholder="Search experiments..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              aria-label="Search experiments by name or parameters"
              inputProps={{ tabIndex: 0 }}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Search />
                  </InputAdornment>
                ),
              }}
              sx={{ flexGrow: 1 }}
            />
            
            <Tooltip title="Filter experiments">
              <IconButton
                onClick={() => console.log('Filter menu would open here')}
                aria-label="Open experiment filter menu"
                tabIndex={0}
              >
                <FilterList />
              </IconButton>
            </Tooltip>
          </Box>

          {selectedExperiments.length > 0 && (
            <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
              <Typography variant="body2" color="text.secondary">
                {selectedExperiments.length} selected
              </Typography>
              <Button
                size="small"
                startIcon={<Compare />}
                onClick={handleCompareSelected}
                disabled={selectedExperiments.length < 2}
                aria-label={`Compare ${selectedExperiments.length} selected experiments`}
                tabIndex={0}
              >
                Compare
              </Button>
              <Button
                size="small"
                startIcon={<Download />}
                onClick={() => console.log('Export selected')}
                aria-label={`Export ${selectedExperiments.length} selected experiments`}
                tabIndex={0}
              >
                Export
              </Button>
              <Button
                size="small"
                startIcon={<Delete />}
                color="error"
                onClick={handleBulkDelete}
                aria-label={`Delete ${selectedExperiments.length} selected experiments`}
                tabIndex={0}
              >
                Delete
              </Button>
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Experiments Table */}
      <Card>
        <CardContent sx={{ p: 2 }}>
          <Box sx={{ height: 600, width: '100%' }}>
            <DataGrid
              rows={filteredExperiments.map((experiment) => ({
                id: experiment.id,
                jobId: experiment.jobId,
                notes: experiment.notes,
                parameters: experiment.parameters,
                executionTime: experiment.executionTime,
                timestamp: experiment.timestamp,
                bestMetric: getBestMetric(experiment.metrics),
              }))}
              columns={[
                {
                  field: 'select',
                  headerName: '',
                  width: 60,
                  sortable: false,
                  disableColumnMenu: true,
                  renderHeader: () => (
                    <Checkbox
                      checked={selectedExperiments.length === filteredExperiments.length && filteredExperiments.length > 0}
                      indeterminate={selectedExperiments.length > 0 && selectedExperiments.length < filteredExperiments.length}
                      onChange={(e) => handleSelectAll(e.target.checked)}
                      aria-label="Select all experiments"
                      tabIndex={0}
                    />
                  ),
                  renderCell: (params: GridRenderCellParams) => (
                    <Checkbox
                      checked={selectedExperiments.includes(params.row.id)}
                      onChange={(e) => handleSelectExperiment(params.row.id, e.target.checked)}
                      aria-label={`Select experiment ${params.row.jobId}`}
                      tabIndex={0}
                    />
                  ),
                },
                {
                  field: 'jobId',
                  headerName: 'Experiment',
                  width: 200,
                  flex: 1,
                  renderCell: (params: GridRenderCellParams) => (
                    <Box>
                      <Typography variant="subtitle2">
                        {params.row.jobId}
                      </Typography>
                      <Typography variant="body2" color="text.secondary" noWrap>
                        {params.row.notes || 'No description'}
                      </Typography>
                    </Box>
                  ),
                },
                {
                  field: 'algorithms',
                  headerName: 'Algorithms',
                  width: 200,
                  renderCell: (params: GridRenderCellParams) => (
                    <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                      {params.row.parameters.methods.map((method: string) => (
                        <Chip
                          key={method}
                          label={method.toUpperCase()}
                          size="small"
                          color={method === params.row.bestMetric.method ? 'primary' : 'default'}
                        />
                      ))}
                    </Box>
                  ),
                },
                {
                  field: 'bestScore',
                  headerName: 'Best Score',
                  width: 150,
                  renderCell: (params: GridRenderCellParams) => (
                    <Box>
                      <Typography variant="body2">
                        {params.row.bestMetric.score.toFixed(3)}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {params.row.bestMetric.method}
                      </Typography>
                    </Box>
                  ),
                },
                {
                  field: 'executionTime',
                  headerName: 'Execution Time',
                  width: 150,
                  renderCell: (params: GridRenderCellParams) => (
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Speed fontSize="small" sx={{ mr: 0.5, color: 'text.secondary' }} />
                      <Typography variant="body2">
                        {params.row.executionTime?.toFixed(1)}s
                      </Typography>
                    </Box>
                  ),
                },
                {
                  field: 'timestamp',
                  headerName: 'Date',
                  width: 150,
                  renderCell: (params: GridRenderCellParams) => (
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <CalendarToday fontSize="small" sx={{ mr: 0.5, color: 'text.secondary' }} />
                      <Typography variant="body2">
                        {formatDate(params.row.timestamp)}
                      </Typography>
                    </Box>
                  ),
                },
                {
                  field: 'actions',
                  headerName: 'Actions',
                  width: 180,
                  sortable: false,
                  disableColumnMenu: true,
                  renderCell: (params: GridRenderCellParams) => (
                    <Box sx={{ display: 'flex', gap: 0.5 }}>
                      <Tooltip title="Load experiment">
                        <IconButton
                          size="small"
                          onClick={() => handleLoadExperiment(params.row.id)}
                          aria-label={`Load experiment ${params.row.jobId}`}
                          tabIndex={0}
                        >
                          <Visibility />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="More actions">
                        <IconButton
                          size="small"
                          onClick={(e) => setActionMenu({ element: e.currentTarget, experimentId: params.row.id })}
                          aria-label={`More actions for experiment ${params.row.jobId}`}
                          tabIndex={0}
                        >
                          <MoreVert />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  ),
                },
              ] as GridColDef[]}
              initialState={{
                pagination: {
                  paginationModel: { pageSize: 50, page: 0 },
                },
              }}
              pageSizeOptions={[50, 100, 500]}
              checkboxSelection={false}
              disableRowSelectionOnClick
              density="compact"
              sx={{
                '& .MuiDataGrid-cell': {
                  borderRight: '1px solid #e0e0e0',
                },
                '& .MuiDataGrid-columnHeaders': {
                  backgroundColor: '#f5f5f5',
                  borderBottom: '2px solid #e0e0e0',
                },
              }}
            />
          </Box>

          {filteredExperiments.length === 0 && (
            <Box sx={{ p: 4, textAlign: 'center' }}>
              <History sx={{ fontSize: 64, color: 'grey.400', mb: 2 }} />
              <Typography variant="h6" color="text.secondary" gutterBottom>
                No experiments found
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {searchTerm ? 'Try adjusting your search terms' : 'Run your first clustering analysis to build experiment history'}
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Action Menu */}
      <Menu
        anchorEl={actionMenu?.element}
        open={Boolean(actionMenu)}
        onClose={() => setActionMenu(null)}
      >
        <MenuItem onClick={() => {
          if (actionMenu) {
            handleEditNotes(actionMenu.experimentId);
            setActionMenu(null);
          }
        }}>
          <Edit sx={{ mr: 1 }} />
          Edit Notes
        </MenuItem>
        <MenuItem onClick={() => {
          console.log('Star experiment');
          setActionMenu(null);
        }}>
          <Star sx={{ mr: 1 }} />
          Add to Favorites
        </MenuItem>
        <MenuItem onClick={() => {
          console.log('Export experiment');
          setActionMenu(null);
        }}>
          <Download sx={{ mr: 1 }} />
          Export
        </MenuItem>
        <Divider />
        <MenuItem 
          onClick={() => {
            if (actionMenu) {
              setDeleteDialog(actionMenu.experimentId);
              setActionMenu(null);
            }
          }}
          sx={{ color: 'error.main' }}
        >
          <Delete sx={{ mr: 1 }} />
          Delete
        </MenuItem>
      </Menu>

      {/* Delete Confirmation Dialog */}
      <Dialog open={Boolean(deleteDialog)} onClose={() => setDeleteDialog(null)}>
        <DialogTitle>Delete Experiment</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete this experiment? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialog(null)} tabIndex={0}>Cancel</Button>
          <Button 
            onClick={() => deleteDialog && handleDeleteExperiment(deleteDialog)}
            color="error"
            tabIndex={0}
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>

      {/* Edit Notes Dialog */}
      <Dialog open={Boolean(editDialog)} onClose={() => setEditDialog(null)} maxWidth="sm" fullWidth>
        <DialogTitle>Edit Experiment Notes</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            multiline
            rows={4}
            value={editNotes}
            onChange={(e) => setEditNotes(e.target.value)}
            placeholder="Add notes about this experiment..."
            inputProps={{ tabIndex: 0 }}
            aria-label="Edit experiment notes"
            sx={{ mt: 1 }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditDialog(null)} tabIndex={0}>Cancel</Button>
          <Button 
            onClick={() => {
              console.log('Save notes:', editNotes);
              setEditDialog(null);
            }}
            tabIndex={0}
          >
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default HistoryPanel;
