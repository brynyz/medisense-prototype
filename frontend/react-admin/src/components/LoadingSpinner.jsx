import React from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { tokens } from '../theme';

const LoadingSpinner = ({ message = "Loading..." }) => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100vh',
        backgroundColor: colors.primary[400],
        gap: 2,
      }}
    >
      <CircularProgress 
        size={60} 
        sx={{ 
          color: colors.greenAccent[500] 
        }} 
      />
      <Typography 
        variant="h6" 
        sx={{ 
          color: colors.grey[100],
          textAlign: 'center'
        }}
      >
        {message}
      </Typography>
    </Box>
  );
};

export default LoadingSpinner;
