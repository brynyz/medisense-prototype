import { useState, useEffect } from 'react';
import { 
  Box, 
  Button, 
  Typography, 
  useTheme, 
  Avatar, 
  Card, 
  CardContent, 
  Grid, 
  Divider,
  IconButton,
  Chip
} from "@mui/material";
import { tokens } from "../../theme";
import Header from "../../components/Header";
import EditIcon from "@mui/icons-material/Edit";
import EmailIcon from "@mui/icons-material/Email";
import PersonIcon from "@mui/icons-material/Person";
import AdminPanelSettingsIcon from "@mui/icons-material/AdminPanelSettings";
import CalendarTodayIcon from "@mui/icons-material/CalendarToday";
import ProfileEdit from "./ProfileEdit";
import axios from 'axios';

const Profile = () => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  const [isEditing, setIsEditing] = useState(false);
  const [userData, setUserData] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch user profile data
  const fetchUserData = async () => {
    try {
      setLoading(true);
      const token = localStorage.getItem('token');
      
      const response = await axios.get(`${process.env.REACT_APP_API_URL}/api/auth/user/`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        }
      });
      
      setUserData(response.data);
    } catch (error) {
      console.error('Error fetching user data:', error);
      setError('Failed to load profile data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchUserData();
  }, []);

  const handleEditComplete = () => {
    setIsEditing(false);
    fetchUserData(); // Refresh data after edit
  };

  if (loading) {
    return (
      <Box m="20px">
        <Header title="PROFILE" subtitle="Loading profile information..." />
        <Typography>Loading...</Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Box m="20px">
        <Header title="PROFILE" subtitle="Error loading profile" />
        <Typography color="error">{error}</Typography>
      </Box>
    );
  }

  if (isEditing) {
    return (
      <ProfileEdit 
        userData={userData} 
        onCancel={() => setIsEditing(false)}
        onSave={handleEditComplete}
      />
    );
  }

  return (
    <Box m="20px">
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb="20px">
        <Header title="PROFILE" subtitle="View and manage your account information" />
        <Button
          onClick={() => setIsEditing(true)}
          variant="contained"
          startIcon={<EditIcon />}
          sx={{
            backgroundColor: colors.blueAccent[600],
            color: colors.grey[100],
            fontSize: "14px",
            fontWeight: "bold",
            padding: "10px 20px",
            "&:hover": {
              backgroundColor: colors.blueAccent[700],
            },
          }}
        >
          Edit Profile
        </Button>
      </Box>

      <Grid container spacing={3}>
        {/* Profile Card */}
        <Grid item xs={12} md={4}>
          <Card
            sx={{
              backgroundColor: colors.primary[400],
              borderRadius: "12px",
              boxShadow: "0 4px 20px rgba(0,0,0,0.1)",
            }}
          >
            <CardContent sx={{ textAlign: "center", p: 3 }}>
              {/* Profile Avatar */}
              <Avatar
                src={userData.profile_image}
                sx={{
                  width: 120,
                  height: 120,
                  margin: "0 auto 20px",
                  border: `4px solid ${colors.blueAccent[500]}`,
                  fontSize: "48px",
                  backgroundColor: colors.blueAccent[600],
                }}
              >
                {userData.first_name?.[0] || userData.username?.[0] || 'U'}
              </Avatar>

              {/* Name */}
              <Typography
                variant="h3"
                color={colors.grey[100]}
                fontWeight="bold"
                mb={1}
              >
                {userData.first_name && userData.last_name 
                  ? `${userData.first_name} ${userData.last_name}`
                  : userData.username
                }
              </Typography>

              {/* Role */}
              <Chip
                label={userData.is_staff ? "Administrator" : "User"}
                sx={{
                  backgroundColor: userData.is_staff 
                    ? colors.greenAccent[600] 
                    : colors.blueAccent[600],
                  color: colors.grey[100],
                  fontWeight: "bold",
                  mb: 2,
                }}
                icon={userData.is_staff ? <AdminPanelSettingsIcon /> : <PersonIcon />}
              />

              {/* Join Date */}
              <Box display="flex" alignItems="center" justifyContent="center" mt={2}>
                <CalendarTodayIcon sx={{ color: colors.grey[300], mr: 1 }} />
                <Typography variant="body2" color={colors.grey[300]}>
                  Joined {new Date(userData.date_joined).toLocaleDateString('en-US', {
                    year: 'numeric',
                    month: 'long',
                    day: 'numeric'
                  })}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Account Information */}
        <Grid item xs={12} md={8}>
          <Card
            sx={{
              backgroundColor: colors.primary[400],
              borderRadius: "12px",
              boxShadow: "0 4px 20px rgba(0,0,0,0.1)",
            }}
          >
            <CardContent sx={{ p: 3 }}>
              <Typography
                variant="h4"
                color={colors.grey[100]}
                fontWeight="bold"
                mb={3}
              >
                Account Information
              </Typography>

              <Grid container spacing={3}>
                {/* Personal Information */}
                <Grid item xs={12} sm={6}>
                  <Box mb={3}>
                    <Typography
                      variant="h6"
                      color={colors.blueAccent[500]}
                      fontWeight="bold"
                      mb={1}
                    >
                      Personal Information
                    </Typography>
                    <Divider sx={{ backgroundColor: colors.grey[600], mb: 2 }} />
                    
                    <Box mb={2}>
                      <Typography variant="body2" color={colors.grey[300]}>
                        First Name
                      </Typography>
                      <Typography variant="body1" color={colors.grey[100]}>
                        {userData.first_name || 'Not provided'}
                      </Typography>
                    </Box>

                    <Box mb={2}>
                      <Typography variant="body2" color={colors.grey[300]}>
                        Last Name
                      </Typography>
                      <Typography variant="body1" color={colors.grey[100]}>
                        {userData.last_name || 'Not provided'}
                      </Typography>
                    </Box>

                    <Box mb={2}>
                      <Typography variant="body2" color={colors.grey[300]}>
                        Username
                      </Typography>
                      <Typography variant="body1" color={colors.grey[100]}>
                        {userData.username}
                      </Typography>
                    </Box>
                  </Box>
                </Grid>

                {/* Contact Information */}
                <Grid item xs={12} sm={6}>
                  <Box mb={3}>
                    <Typography
                      variant="h6"
                      color={colors.blueAccent[500]}
                      fontWeight="bold"
                      mb={1}
                    >
                      Contact Information
                    </Typography>
                    <Divider sx={{ backgroundColor: colors.grey[600], mb: 2 }} />
                    
                    <Box mb={2} display="flex" alignItems="center">
                      <EmailIcon sx={{ color: colors.grey[300], mr: 1 }} />
                      <Box>
                        <Typography variant="body2" color={colors.grey[300]}>
                          Email Address
                        </Typography>
                        <Typography variant="body1" color={colors.grey[100]}>
                          {userData.email || 'Not provided'}
                        </Typography>
                      </Box>
                    </Box>
                  </Box>
                </Grid>

                {/* Account Status */}
                <Grid item xs={12}>
                  <Box>
                    <Typography
                      variant="h6"
                      color={colors.blueAccent[500]}
                      fontWeight="bold"
                      mb={1}
                    >
                      Account Status
                    </Typography>
                    <Divider sx={{ backgroundColor: colors.grey[600], mb: 2 }} />
                    
                    <Grid container spacing={2}>
                      <Grid item xs={12} sm={4}>
                        <Box textAlign="center" p={2}>
                          <Typography variant="h6" color={colors.greenAccent[500]}>
                            {userData.is_active ? 'Active' : 'Inactive'}
                          </Typography>
                          <Typography variant="body2" color={colors.grey[300]}>
                            Account Status
                          </Typography>
                        </Box>
                      </Grid>
                      
                      <Grid item xs={12} sm={4}>
                        <Box textAlign="center" p={2}>
                          <Typography variant="h6" color={colors.blueAccent[500]}>
                            {userData.is_staff ? 'Admin' : 'Standard'}
                          </Typography>
                          <Typography variant="body2" color={colors.grey[300]}>
                            User Type
                          </Typography>
                        </Box>
                      </Grid>

                      <Grid item xs={12} sm={4}>
                        <Box textAlign="center" p={2}>
                          <Typography variant="h6" color={colors.grey[100]}>
                            {new Date(userData.last_login).toLocaleDateString() || 'Never'}
                          </Typography>
                          <Typography variant="body2" color={colors.grey[300]}>
                            Last Login
                          </Typography>
                        </Box>
                      </Grid>
                    </Grid>
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Profile;
