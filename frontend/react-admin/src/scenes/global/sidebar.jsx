import { useState, useEffect } from "react";
import { ProSidebar, Menu, MenuItem } from "react-pro-sidebar";
import { Box, IconButton, Typography, useTheme } from "@mui/material";
import { Link, useNavigate } from "react-router-dom";
import axios from 'axios'; // Import axios
import "react-pro-sidebar/dist/css/styles.css";
import { tokens } from "../../theme";
import HomeOutlinedIcon from "@mui/icons-material/HomeOutlined";
import PeopleOutlinedIcon from "@mui/icons-material/PeopleOutlined";
import CalendarTodayOutlinedIcon from "@mui/icons-material/CalendarTodayOutlined";
import HelpOutlineOutlinedIcon from "@mui/icons-material/HelpOutlineOutlined";
import DataExplorationIcon from '@mui/icons-material/DataExploration';
import MenuOutlinedIcon from "@mui/icons-material/MenuOutlined";
import LocalHospitalIcon from "@mui/icons-material/LocalHospital";
import AssistantIcon from '@mui/icons-material/Assistant';
import DatasetIcon from '@mui/icons-material/Dataset';
import LogoutIcon from '@mui/icons-material/Logout';
import SettingsIcon from '@mui/icons-material/Settings';

const Item = ({ title, to, icon, selected, setSelected }) => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  return (
    <MenuItem
      active={selected === title}
      style={{
        color: colors.grey[100],
      }}
      onClick={() => setSelected(title)}
      icon={icon}
    >
      <Typography>{title}</Typography>
      <Link to={to} />
    </MenuItem>
  );
};

const Sidebar = ({ isSidebar, setIsSidebar, user }) => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [selected, setSelected] = useState("Dashboard");
  const navigate = useNavigate();
  const [userData, setUserData] = useState({});

  useEffect(() => {
    const fetchUserData = async () => {
      try {
        const token = localStorage.getItem('token');
        if (!token) {
          console.log('No token found, redirecting to login');
          navigate('/login');
          return;
        }

        // console.log('Fetching user data from API...');
        // console.log('API URL:', process.env.REACT_APP_API_URL);
        // console.log('Token:', token?.substring(0, 20) + '...');
        
        // Test API connectivity first
        const apiUrl = `${process.env.REACT_APP_API_URL}/api/auth/user/`;
        // console.log('Full API URL:', apiUrl);
        
        const response = await axios.get(apiUrl, {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
          timeout: 10000 // 10 second timeout
        });
        
        // console.log('User data received:', response.data);
        setUserData(response.data);
      } catch (error) {
        console.error('Error fetching user data:', error);
        
        if (error.code === 'ECONNABORTED') {
          console.error('Request timed out - Django server might not be running');
        } else if (error.message === 'Network Error') {
          console.error('Network error - Check if Django server is running on the correct port');
        }
        
        console.log('Error details:', error.response?.data);
        console.log('Error status:', error.response?.status);
        console.log('Error message:', error.message);
        
        // Fallback to localStorage user data
        const localUser = localStorage.getItem('user');
        if (localUser) {
          try {
            const userData = JSON.parse(localUser);
            console.log('Using localStorage user data:', userData);
            setUserData(userData);
          } catch (parseError) {
            console.error('Error parsing localStorage user data:', parseError);
            setUserData({ username: 'User', is_staff: false });
          }
        } else {
          console.log('No localStorage user data found, using default');
          setUserData({ username: 'User', is_staff: false });
        }
        
        if (error.response?.status === 401) {
          console.log('Token expired, clearing storage and redirecting');
          localStorage.removeItem('token');
          localStorage.removeItem('refresh_token');
          localStorage.removeItem('user');
          navigate('/login');
        }
      }
    };

    fetchUserData();
  }, [navigate]);

  const handleLogout = async () => {
    try {
      const token = localStorage.getItem('token');
      
      if (token) {
        await axios.post(`${process.env.REACT_APP_API_URL}/api/auth/logout-user/`, {}, {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          }
        });
      }
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      // Clear local storage regardless of API call success
      localStorage.removeItem('token');
      localStorage.removeItem('refresh_token');
      localStorage.removeItem('user');
      navigate("/login");
    }
  };

  // Update parent component when collapsed state changes
  const handleToggleCollapse = () => {
    const newCollapsedState = !isCollapsed;
    setIsCollapsed(newCollapsedState);
    if (setIsSidebar) {
      setIsSidebar(!newCollapsedState); // isSidebar is opposite of isCollapsed
    }
  };

  return (
    <Box
      sx={{
        "& .pro-sidebar-inner": {
          background: `${colors.primary[400]} !important`,
        },
        "& .pro-icon-wrapper": {
          backgroundColor: "transparent !important",
        },
        "& .pro-inner-item": {
          padding: "5px 35px 5px 20px !important",
        },
        "& .pro-inner-item:hover": {
          color: "#868dfb !important",
        },
        "& .pro-menu-item.active": {
          color: "#6870fa !important",
        },
        height: "100vh",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <ProSidebar collapsed={isCollapsed} style={{ height: "100%", flex: 1 }}>
        <Menu iconShape="square" style={{ height: "100%", display: "flex", flexDirection: "column" }}>
          {/* LOGO AND MENU ICON */}
          <MenuItem
            onClick={handleToggleCollapse}
            icon={isCollapsed ? <MenuOutlinedIcon /> : undefined}
            style={{
              margin: "10px 0 20px 0",
              color: colors.grey[100],
            }}
          >
            {!isCollapsed && (
              <Box
                display="flex"
                justifyContent="space-between"
                alignItems="center"
                ml="15px"
              >
                <Typography variant="h3" color={colors.grey[100]}>
                  MEDISENSE
                </Typography>
                <IconButton onClick={handleToggleCollapse}>
                  <MenuOutlinedIcon />
                </IconButton>
              </Box>
            )}
          </MenuItem>

          {!isCollapsed && (
            <Box mb="25px">
              <Box display="flex" justifyContent="center" alignItems="center">
                <img
                  alt="profile-user"
                  width="100px"
                  height="100px"
                  src={userData.profile_image}
                  style={{ cursor: "pointer", borderRadius: "50%" }}
                />
              </Box>
              <Box textAlign="center">
                <Typography
                  variant="h2"
                  color={colors.grey[100]}
                  fontWeight="bold"
                  sx={{ m: "10px 0 0 0" }}
                >
                  {userData && userData.first_name 
                    ? `${userData.first_name}`
                    : userData?.username || 'Loading...'
                  }
                </Typography>
                <Typography variant="h5" color={colors.greenAccent[500]}>
                  {userData.role || (userData.is_staff ? 'Administrator' : 'User')}
                </Typography>
              </Box>
            </Box>
          )}

          {/* Main Menu Items */}
          <Box 
            paddingLeft={isCollapsed ? undefined : "10%"} 
            flex={1}
            className="custom-scroll"
            sx={{ 
              overflowY: "auto",
              paddingRight: "10px",
            }}
          >
            <Item
              title="Dashboard"
              to="/dashboard"
              icon={<HomeOutlinedIcon />}
              selected={selected}
              setSelected={setSelected}
            />
            <Item
              title="Symptoms"
              to="/app/symptoms"
              icon={<LocalHospitalIcon />}
              selected={selected}
              setSelected={setSelected}
            />
            <Item
              title="Data Preprocessing"
              to="/app/datapreprocessing"
              icon={<DatasetIcon />}
              selected={selected}
              setSelected={setSelected}
            />
            <Item
              title="Model Training"
              to="/app/modeltraining"
              icon={<DataExplorationIcon />}
              selected={selected}
              setSelected={setSelected}
            />

            <Typography
              variant="h6"
              color={colors.grey[300]}
              sx={{ m: "15px 0 5px 20px" }}
            >
              Help
            </Typography>
            <Item
              title="Calendar"
              to="/app/calendar"
              icon={<CalendarTodayOutlinedIcon />}
              selected={selected}
              setSelected={setSelected}
            />
            <Item
              title="Smart Help"
              to="/app/help"
              icon={<AssistantIcon />}
              selected={selected}
              setSelected={setSelected}
            />
            <Item
              title="FAQ Page"
              to="/app/faq"
              icon={<HelpOutlineOutlinedIcon />}
              selected={selected}
              setSelected={setSelected}
            />
            <Item
              title="Settings"
              to="/app/profile"
              icon={<SettingsIcon />}
              selected={selected}
              setSelected={setSelected}
            />  
            {/* <Item
              title="Map"
              to="/app/heatmap"
              icon={<SettingsIcon />}
              selected={selected}
              setSelected={setSelected}
            /> */}
          </Box>

          {/* Logout Button at Bottom */}
          <Box 
            paddingLeft={isCollapsed ? undefined : "10%"} 
            sx={{ 
              marginTop: "auto", 
              paddingBottom: "20px",
              paddingTop: "200px"
            }}
          >
            <MenuItem
              icon={<LogoutIcon />}
              style={{
                color: colors.redAccent[400],
                "&:hover": {
                  color: colors.redAccent[300],
                }
              }}
              onClick={handleLogout}
            >
              <Typography>Logout</Typography>
            </MenuItem>
          </Box>  
        </Menu>
      </ProSidebar>
    </Box>
  );
};

export default Sidebar;