import React from 'react';
import { Box, Paper, Typography, Button, Alert } from '@mui/material';
import { useAuth } from '../contexts/AuthContext';
import { getToken } from '../services/api';

const AuthDebugPanel = () => {
  const auth = useAuth();
  const token = getToken();

  const handleLogDetails = () => {
    console.group('🔍 Authentication Debug Details');
    console.log('Auth Context State:', {
      isAuthenticated: auth.isAuthenticated,
      isLoading: auth.isLoading,
      user: auth.user,
      error: auth.error
    });
    console.log('Token from localStorage:', token);
    console.log('Token exists:', !!token);
    if (token) {
      console.log('Token length:', token.length);
      console.log('Token starts with:', token.substring(0, 10));
    }
    console.log('All localStorage keys:', Object.keys(localStorage));
    console.log('localStorage token:', localStorage.getItem('token'));
    console.log('localStorage refresh_token:', localStorage.getItem('refresh_token'));
    console.log('localStorage user:', localStorage.getItem('user'));
    console.groupEnd();
  };

  const handleTestAPI = async () => {
    try {
      console.log('Testing API call...');
      const response = await fetch('https://medisense-backend-ml3h5.ondigitalocean.app/api/patients/symptoms/', {
        headers: {
          'Authorization': token ? `Bearer ${token}` : 'No token',
          'Content-Type': 'application/json'
        }
      });
      console.log('API Response status:', response.status);
      console.log('API Response headers:', Object.fromEntries(response.headers.entries()));
      const data = await response.text();
      console.log('API Response data:', data);
    } catch (error) {
      console.error('API Test failed:', error);
    }
  };

  return (
    <Paper sx={{ p: 2, m: 2, backgroundColor: '#f5f5f5' }}>
      <Typography variant="h6" gutterBottom color="primary">
        🔍 Authentication Debug Panel
      </Typography>
      
      <Box sx={{ mb: 2 }}>
        <Typography><strong>Is Authenticated:</strong> {String(auth.isAuthenticated)}</Typography>
        <Typography><strong>Is Loading:</strong> {String(auth.isLoading)}</Typography>
        <Typography><strong>Has Token:</strong> {String(!!token)}</Typography>
        <Typography><strong>User:</strong> {auth.user ? auth.user.username || 'No username' : 'No user'}</Typography>
        <Typography><strong>Error:</strong> {auth.error || 'None'}</Typography>
      </Box>

      {!auth.isAuthenticated && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          User is not authenticated! This is why API calls are failing.
        </Alert>
      )}

      {!token && (
        <Alert severity="error" sx={{ mb: 2 }}>
          No token found in localStorage! User needs to log in.
        </Alert>
      )}

      <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
        <Button variant="contained" onClick={handleLogDetails} size="small">
          Log Auth Details
        </Button>
        <Button variant="outlined" onClick={handleTestAPI} size="small">
          Test API Call
        </Button>
      </Box>
    </Paper>
  );
};

export default AuthDebugPanel;
