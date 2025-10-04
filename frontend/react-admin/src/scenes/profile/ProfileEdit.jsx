import { useState, useRef } from 'react';
import {
  Box,
  Button,
  TextField,
  Typography,
  useTheme,
  Avatar,
  Card,
  CardContent,
  Grid,
  IconButton,
  Alert,
  CircularProgress,
  Divider,
  Tooltip,
  Paper
} from "@mui/material";
import { tokens } from "../../theme";
import Header from "../../components/Header";
import SaveIcon from "@mui/icons-material/Save";
import CancelIcon from "@mui/icons-material/Cancel";
import PhotoCameraIcon from "@mui/icons-material/PhotoCamera";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import InfoIcon from "@mui/icons-material/Info";
import EmailIcon from "@mui/icons-material/Email";
import axios from 'axios';

const ProfileEdit = ({ userData, onCancel, onSave }) => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  const fileInputRef = useRef(null);
  
  const [formData, setFormData] = useState({
    first_name: userData.first_name || '',
    last_name: userData.last_name || '',
    email: userData.email || '',
    role: userData.role || 'User',
  });
  
  const [profileImage, setProfileImage] = useState(userData.profile_image || null);
  const [imageFile, setImageFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [errors, setErrors] = useState({});

  // Handle form input changes
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    
    // Clear field error when user starts typing
    if (errors[name]) {
      setErrors(prev => ({
        ...prev,
        [name]: ''
      }));
    }
  };

  // Handle image upload
  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      // Validate file type
      if (!file.type.startsWith('image/')) {
        setError('Please select a valid image file');
        return;
      }
      
      // Validate file size (800x400px max as per template)
      if (file.size > 5 * 1024 * 1024) {
        setError('Image size must be less than 5MB');
        return;
      }
      
      setImageFile(file);
      
      // Create preview URL
      const reader = new FileReader();
      reader.onload = (e) => {
        setProfileImage(e.target.result);
      };
      reader.readAsDataURL(file);
      
      setError(null);
    }
  };

  // Validate form
  const validateForm = () => {
    const newErrors = {};
    
    if (!formData.first_name.trim()) {
      newErrors.first_name = 'First name is required';
    }
    
    if (!formData.last_name.trim()) {
      newErrors.last_name = 'Last name is required';
    }
    
    if (!formData.email.trim()) {
      newErrors.email = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = 'Email is invalid';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }
    
    setLoading(true);
    setError(null);
    setSuccess(null);
    
    try {
      const token = localStorage.getItem('token');
      
      // Create FormData for multipart/form-data
      const formDataToSend = new FormData();
      formDataToSend.append('first_name', formData.first_name);
      formDataToSend.append('last_name', formData.last_name);
      formDataToSend.append('email', formData.email);
      formDataToSend.append('username', userData.username); // Keep existing username
      
      // Add image if selected
      if (imageFile) {
        formDataToSend.append('profile_image', imageFile);
      }
      
      const response = await axios.put(
        `${process.env.REACT_APP_API_URL}/api/auth/profile/update/`,
        formDataToSend,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'multipart/form-data',
          }
        }
      );
      
      setSuccess('Profile updated successfully!');
      
      // Update localStorage user data
      const updatedUser = { ...userData, ...response.data };
      localStorage.setItem('user', JSON.stringify(updatedUser));
      
      // Call onSave after a short delay to show success message
      setTimeout(() => {
        onSave();
      }, 1500);
      
    } catch (error) {
      console.error('Error updating profile:', error);
      
      if (error.response?.data) {
        // Handle validation errors from backend
        if (typeof error.response.data === 'object') {
          setErrors(error.response.data);
        } else {
          setError(error.response.data.detail || 'Failed to update profile');
        }
      } else {
        setError('Failed to update profile. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box m="20px">
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb="20px">
        <Header title="EDIT PROFILE" subtitle="Update your photo and personal details here." />
        
        {/* Action Buttons */}
        <Box display="flex" gap={2}>
          <Button
            onClick={onCancel}
            variant="outlined"
            startIcon={<CancelIcon />}
            sx={{
              borderColor: colors.grey[400],
              color: colors.grey[100],
              backgroundColor: colors.primary[500],
              "&:hover": {
                borderColor: colors.grey[300],
                backgroundColor: colors.primary[400],
              },
            }}
          >
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            variant="contained"
            startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <SaveIcon />}
            disabled={loading}
            sx={{
              backgroundColor: colors.blueAccent[600],
              color: colors.grey[100],
              "&:hover": {
                backgroundColor: colors.blueAccent[700],
              },
              "&:disabled": {
                backgroundColor: colors.grey[600],
              },
            }}
          >
            {loading ? 'Saving...' : 'Save'}
          </Button>
        </Box>
      </Box>

      {/* Alerts */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}
      
      {success && (
        <Alert severity="success" sx={{ mb: 3 }}>
          {success}
        </Alert>
      )}

      {/* Main Content Card */}
      <Card
        sx={{
          backgroundColor: colors.primary[400],
          borderRadius: "12px",
          boxShadow: "0 4px 20px rgba(0,0,0,0.1)",
        }}
      >
        <CardContent sx={{ p: 4 }}>
          {/* Personal Info Section Header */}
          <Typography
            variant="h4"
            color={colors.grey[100]}
            fontWeight="bold"
            mb={1}
          >
            Personal info
          </Typography>
          <Typography
            variant="body1"
            color={colors.grey[300]}
            mb={4}
          >
            Update your photo and personal details here.
          </Typography>

          <Box component="form" onSubmit={handleSubmit}>
            {/* Name Fields */}
            <Box mb={4}>
              <Typography
                variant="body1"
                color={colors.grey[100]}
                fontWeight="500"
                mb={2}
              >
                Name <span style={{ color: colors.redAccent[400] }}>*</span>
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    placeholder="First name"
                    name="first_name"
                    value={formData.first_name}
                    onChange={handleChange}
                    error={!!errors.first_name}
                    helperText={errors.first_name}
                    variant="outlined"
                    sx={{
                      '& .MuiOutlinedInput-root': {
                        backgroundColor: colors.primary[500],
                        '& fieldset': {
                          borderColor: colors.grey[600],
                        },
                        '&:hover fieldset': {
                          borderColor: colors.grey[500],
                        },
                        '&.Mui-focused fieldset': {
                          borderColor: colors.blueAccent[500],
                        },
                      },
                      '& .MuiInputBase-input': {
                        color: colors.grey[100],
                        '&::placeholder': {
                          color: colors.grey[400],
                          opacity: 1,
                        },
                      },
                      '& .MuiFormHelperText-root': {
                        color: colors.redAccent[400],
                      },
                    }}
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    placeholder="Last name"
                    name="last_name"
                    value={formData.last_name}
                    onChange={handleChange}
                    error={!!errors.last_name}
                    helperText={errors.last_name}
                    variant="outlined"
                    sx={{
                      '& .MuiOutlinedInput-root': {
                        backgroundColor: colors.primary[500],
                        '& fieldset': {
                          borderColor: colors.grey[600],
                        },
                        '&:hover fieldset': {
                          borderColor: colors.grey[500],
                        },
                        '&.Mui-focused fieldset': {
                          borderColor: colors.blueAccent[500],
                        },
                      },
                      '& .MuiInputBase-input': {
                        color: colors.grey[100],
                        '&::placeholder': {
                          color: colors.grey[400],
                          opacity: 1,
                        },
                      },
                      '& .MuiFormHelperText-root': {
                        color: colors.redAccent[400],
                      },
                    }}
                  />
                </Grid>
              </Grid>
            </Box>

            {/* Email Field */}
            <Box mb={4}>
              <Typography
                variant="body1"
                color={colors.grey[100]}
                fontWeight="500"
                mb={2}
              >
                Email address <span style={{ color: colors.redAccent[400] }}>*</span>
              </Typography>
              <TextField
                fullWidth
                placeholder="olivia@untitledui.com"
                name="email"
                type="email"
                value={formData.email}
                onChange={handleChange}
                error={!!errors.email}
                helperText={errors.email}
                variant="outlined"
                InputProps={{
                  startAdornment: (
                    <EmailIcon sx={{ color: colors.grey[400], mr: 1 }} />
                  ),
                }}
                sx={{
                  '& .MuiOutlinedInput-root': {
                    backgroundColor: colors.primary[500],
                    '& fieldset': {
                      borderColor: colors.grey[600],
                    },
                    '&:hover fieldset': {
                      borderColor: colors.grey[500],
                    },
                    '&.Mui-focused fieldset': {
                      borderColor: colors.blueAccent[500],
                    },
                  },
                  '& .MuiInputBase-input': {
                    color: colors.grey[100],
                    '&::placeholder': {
                      color: colors.grey[400],
                      opacity: 1,
                    },
                  },
                  '& .MuiFormHelperText-root': {
                    color: colors.redAccent[400],
                  },
                }}
              />
            </Box>

            {/* Photo Upload Section */}
            <Box mb={4}>
              <Box display="flex" alignItems="center" mb={2}>
                <Typography
                  variant="body1"
                  color={colors.grey[100]}
                  fontWeight="500"
                  mr={1}
                >
                  Your photo <span style={{ color: colors.redAccent[400] }}>*</span>
                </Typography>
                <Tooltip title="This will be displayed on your profile." arrow>
                  <InfoIcon sx={{ color: colors.grey[400], fontSize: 18 }} />
                </Tooltip>
              </Box>
              
              <Grid container spacing={3} alignItems="center">
                {/* Current Photo */}
                <Grid item>
                  <Avatar
                    src={profileImage}
                    sx={{
                      width: 80,
                      height: 80,
                      border: `2px solid ${colors.grey[600]}`,
                      fontSize: "32px",
                      backgroundColor: colors.blueAccent[600],
                    }}
                  >
                    {formData.first_name?.[0] || userData.username?.[0] || 'U'}
                  </Avatar>
                </Grid>

                {/* Upload Area */}
                <Grid item xs>
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleImageUpload}
                    accept="image/*"
                    style={{ display: 'none' }}
                  />
                  
                  <Paper
                    onClick={() => fileInputRef.current?.click()}
                    sx={{
                      p: 3,
                      border: `2px dashed ${colors.grey[600]}`,
                      backgroundColor: colors.primary[500],
                      cursor: 'pointer',
                      textAlign: 'center',
                      '&:hover': {
                        borderColor: colors.blueAccent[500],
                        backgroundColor: colors.primary[400],
                      },
                    }}
                  >
                    <CloudUploadIcon 
                      sx={{ 
                        color: colors.blueAccent[500], 
                        fontSize: 32,
                        mb: 1 
                      }} 
                    />
                    <Typography
                      variant="body1"
                      color={colors.grey[100]}
                      fontWeight="500"
                      mb={1}
                    >
                      Click to upload or drag and drop
                    </Typography>
                    <Typography
                      variant="body2"
                      color={colors.grey[400]}
                    >
                      SVG, PNG, JPG or GIF (max. 800x400px)
                    </Typography>
                  </Paper>
                </Grid>
              </Grid>
            </Box>

            {/* Role Field */}
            <Box mb={4}>
              <Typography
                variant="body1"
                color={colors.grey[100]}
                fontWeight="500"
                mb={2}
              >
                Role
              </Typography>
              <TextField
                fullWidth
                placeholder="Product Designer"
                name="role"
                value={formData.role}
                onChange={handleChange}
                variant="outlined"
                sx={{
                  '& .MuiOutlinedInput-root': {
                    backgroundColor: colors.primary[500],
                    '& fieldset': {
                      borderColor: colors.grey[600],
                    },
                    '&:hover fieldset': {
                      borderColor: colors.grey[500],
                    },
                    '&.Mui-focused fieldset': {
                      borderColor: colors.blueAccent[500],
                    },
                  },
                  '& .MuiInputBase-input': {
                    color: colors.grey[100],
                    '&::placeholder': {
                      color: colors.grey[400],
                      opacity: 1,
                    },
                  },
                }}
              />
            </Box>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default ProfileEdit;
