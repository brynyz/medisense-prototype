import React, { useState } from 'react';
import { 
  Box, 
  Button, 
  Typography, 
  Paper, 
  Alert, 
  CircularProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Grid
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { debugAPI, symptomsAPI, authAPI } from '../../services/api';

const APIDebugTool = () => {
  const [results, setResults] = useState({});
  const [loading, setLoading] = useState({});

  const runTest = async (testName, testFunction) => {
    setLoading(prev => ({ ...prev, [testName]: true }));
    try {
      const result = await testFunction();
      setResults(prev => ({
        ...prev,
        [testName]: {
          success: true,
          data: result,
          timestamp: new Date().toLocaleString()
        }
      }));
    } catch (error) {
      setResults(prev => ({
        ...prev,
        [testName]: {
          success: false,
          error: error.message,
          details: error,
          timestamp: new Date().toLocaleString()
        }
      }));
    }
    setLoading(prev => ({ ...prev, [testName]: false }));
  };

  const tests = [
    {
      name: 'environment',
      title: 'Environment Info',
      description: 'Check current environment configuration',
      action: () => debugAPI.getEnvironmentInfo()
    },
    {
      name: 'healthCheck',
      title: 'API Health Check',
      description: 'Test basic API connectivity',
      action: () => debugAPI.healthCheck()
    },
    {
      name: 'authEndpoint',
      title: 'Auth Endpoint Test',
      description: 'Test authentication endpoint availability',
      action: () => debugAPI.testEndpoint('/api/auth/register/', 'OPTIONS')
    },
    {
      name: 'symptomsEndpoint',
      title: 'Symptoms Endpoint Test',
      description: 'Test the failing symptoms endpoint',
      action: () => debugAPI.testEndpoint('/api/patients/symptoms/')
    },
    {
      name: 'symptomsAPI',
      title: 'Full Symptoms API Call',
      description: 'Make actual API call to symptoms endpoint',
      action: () => symptomsAPI.getAll()
    }
  ];

  const renderResult = (result) => {
    if (!result) return null;

    return (
      <Box sx={{ mt: 2 }}>
        <Alert severity={result.success ? 'success' : 'error'} sx={{ mb: 2 }}>
          {result.success ? 'Test passed' : 'Test failed'}
          <Typography variant="caption" display="block">
            {result.timestamp}
          </Typography>
        </Alert>
        
        <Paper sx={{ p: 2, bgcolor: 'background.paper' }}>
          <Typography variant="body2" component="pre" sx={{ 
            whiteSpace: 'pre-wrap', 
            wordBreak: 'break-word',
            fontSize: '0.875rem',
            fontFamily: 'monospace'
          }}>
            {JSON.stringify(result.success ? result.data : result.details, null, 2)}
          </Typography>
        </Paper>
      </Box>
    );
  };

  return (
    <Box sx={{ p: 3, maxWidth: 1200, mx: 'auto' }}>
      <Typography variant="h4" gutterBottom>
        API Debug Tool
      </Typography>
      
      <Typography variant="body1" sx={{ mb: 3, color: 'text.secondary' }}>
        Use this tool to debug API connectivity issues and test individual endpoints.
      </Typography>

      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item>
          <Button
            variant="contained"
            onClick={() => {
              // Run all tests
              tests.forEach(test => {
                setTimeout(() => runTest(test.name, test.action), Math.random() * 1000);
              });
            }}
          >
            Run All Tests
          </Button>
        </Grid>
        <Grid item>
          <Button
            variant="outlined"
            onClick={() => {
              setResults({});
              console.clear();
            }}
          >
            Clear Results
          </Button>
        </Grid>
      </Grid>

      {tests.map((test) => (
        <Accordion key={test.name} sx={{ mb: 2 }}>
          <AccordionSummary
            expandIcon={<ExpandMoreIcon />}
            aria-controls={`${test.name}-content`}
            id={`${test.name}-header`}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, width: '100%' }}>
              <Typography variant="h6">{test.title}</Typography>
              <Typography variant="body2" color="text.secondary" sx={{ flexGrow: 1 }}>
                {test.description}
              </Typography>
              <Button
                size="small"
                variant="outlined"
                onClick={(e) => {
                  e.stopPropagation();
                  runTest(test.name, test.action);
                }}
                disabled={loading[test.name]}
                startIcon={loading[test.name] ? <CircularProgress size={16} /> : null}
              >
                {loading[test.name] ? 'Testing...' : 'Test'}
              </Button>
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            {renderResult(results[test.name])}
          </AccordionDetails>
        </Accordion>
      ))}

      <Paper sx={{ p: 3, mt: 4, bgcolor: 'background.default' }}>
        <Typography variant="h6" gutterBottom>
          Common Issues & Solutions:
        </Typography>
        <Typography variant="body2" paragraph>
          <strong>500 Server Error:</strong> The backend server is experiencing an internal error. 
          Check the Django logs on your DigitalOcean app.
        </Typography>
        <Typography variant="body2" paragraph>
          <strong>CORS Issues:</strong> Make sure your Django backend has proper CORS configuration 
          for your frontend domain.
        </Typography>
        <Typography variant="body2" paragraph>
          <strong>Environment Variables:</strong> Ensure REACT_APP_API_URL is properly set and 
          the React app is restarted after changes.
        </Typography>
      </Paper>
    </Box>
  );
};

export default APIDebugTool;