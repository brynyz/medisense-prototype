import { Box, Button, IconButton, Typography, useTheme } from "@mui/material";
import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { tokens } from "../../theme";
import DownloadOutlinedIcon from "@mui/icons-material/DownloadOutlined";
import Tour from "@mui/icons-material/Tour";
import CoronavirusIcon from '@mui/icons-material/Coronavirus';
import EmailIcon from "@mui/icons-material/Email";
import FullScreenIcon from "@mui/icons-material/Fullscreen";
import PointOfSaleIcon from "@mui/icons-material/PointOfSale";
import PersonAddIcon from "@mui/icons-material/PersonAdd";
import Header from "../../components/Header";
import LineChart from "../../components/LineChart";
import CampusHeatmap from "../../components/CampusHeatmap";
import BarChart from "../../components/BarChart";
import StatBox from "../../components/StatBox";
import ProgressCircle from "../../components/ProgressCircle";
import EnvironmentalStats from "../../components/EnvironmentalStats";
import RecentActivities from "../../components/RecentActivities";

const Dashboard = () => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  const [totalVisits, setTotalVisits] = useState(0);
  const [loading, setLoading] = useState(true);

  // Fetch total visit count
  useEffect(() => {
    const fetchTotalVisits = async () => {
      try {
        setLoading(true);
        const token = localStorage.getItem('token');
        
        if (!token) {
          console.log('No token found');
          return;
        }

        console.log('🔑 Token found:', token ? 'Yes' : 'No');
        console.log('🌐 API URL:', `${process.env.REACT_APP_API_URL}/api/patients/symptoms/`);

        const response = await fetch(`${process.env.REACT_APP_API_URL}/api/patients/symptoms/`, {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
        });

        console.log('📡 Response status:', response.status);

        if (response.ok) {
          const data = await response.json();
          // console.log('📊 Raw API data:', data);
          // console.log('📊 Data type:', typeof data);
          // console.log('📊 Is array:', Array.isArray(data));
          // console.log('📊 Has results:', data.results ? 'Yes' : 'No');
          
          // Count total visits (length of symptoms array)
          const visitCount = Array.isArray(data) ? data.length : (data.results ? data.results.length : 0);
          setTotalVisits(visitCount);
          console.log('✅ Total visits set to:', visitCount);
        } else {
          const errorText = await response.text();
          console.error('❌ Failed to fetch visits:', response.status, errorText);
        }
      } catch (error) {
        console.error('Error fetching total visits:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchTotalVisits();
    
    // Refresh every 30 seconds for live updates
    const interval = setInterval(fetchTotalVisits, 30000);
    
    return () => clearInterval(interval);
  }, []);
  return (
    <Box m="20px">
      {/* HEADER */}
      <Box display="flex" justifyContent="space-between" alignItems="center">
        <Header title="Analytics" subtitle="Campus Clinic Analytics" />

        <Box>
          <Button
            sx={{
              backgroundColor: colors.blueAccent[700],
              color: colors.grey[100],
              fontSize: "14px",
              fontWeight: "bold",
              padding: "10px 20px",
            }}
          >
            <DownloadOutlinedIcon sx={{ mr: "10px" }} />
            Download Reports
          </Button>
        </Box>
      </Box>

      {/* GRID & CHARTS */}
      <Box
        display="grid"
        gridTemplateColumns="repeat(12, 1fr)"
        gridAutoRows="140px"
        gap="10px"
      >
        {/* ROW 1 */}
        <Box
          gridColumn="span 3"
          backgroundColor={colors.primary[400]}
          display="flex"
          alignItems="center"
          justifyContent="center"
        >
          <StatBox
            title="No Visit"
            subtitle="Visit/No Visit Prediction"
            progress="0.65"
            increase="65%"
            icon={
              < Tour
                sx={{ color: colors.greenAccent[600], fontSize: "26px" }}
              />
            }
          />
        </Box>
        <Box
          gridColumn="span 3"
          backgroundColor={colors.primary[400]}
          display="flex"
          alignItems="center"
          justifyContent="center"
        >
          <StatBox
            title="0"
            subtitle="Predicted Visit Count (within ±2 error)"
            progress="0"
            increase="+0%"
            icon={
              <PersonAddIcon
                sx={{ color: colors.greenAccent[600], fontSize: "26px" }}
              />
            }
          />
        </Box>
        <Box
          gridColumn="span 3"
          backgroundColor={colors.primary[400]}
          display="flex"
          alignItems="center"
          justifyContent="center"
        >
          <StatBox
            title="Systemic"
            subtitle="Predicted Dominant Symptom Category"
            progress="0.7"
            increase="70%"
            icon={
              <CoronavirusIcon
                sx={{ color: colors.greenAccent[600], fontSize: "26px" }}
              />
            }
          />
        </Box>
        <Box
          gridColumn="span 3"
          backgroundColor={colors.primary[400]}
          display="flex"
          alignItems="center"
          justifyContent="center"
        >
          <EnvironmentalStats />
        </Box>

        {/* ROW 2 */}
        <Box
          gridColumn="span 8"
          gridRow="span 2"
          backgroundColor={colors.primary[400]}
        >
          <Box
            mt="25px"
            p="0 30px"
            display="flex "
            justifyContent="space-between"
            alignItems="center"
          >
            <Box>
              <Typography
                variant="h5"
                fontWeight="600"
                color={colors.grey[100]}
              >
                Visit Count per Symptom Category
              </Typography>
              <Typography
                variant="h3"
                fontWeight="bold"
                color={colors.greenAccent[500]}
              >
                {loading ? "Loading..." : totalVisits.toLocaleString()}
              </Typography>
            </Box>
            <Box>
              <IconButton>
                <DownloadOutlinedIcon
                  sx={{ fontSize: "26px", color: colors.greenAccent[500] }}
                />
              </IconButton>
            </Box>
          </Box>
          <Box height="250px" m="-20px 0 0 0">
            <LineChart isDashboard={true} />
          </Box>
        </Box>

        <Box
          gridColumn="span 4"
          gridRow="span 2"
          backgroundColor={colors.primary[400]}
          overflow="auto"
        >
          <RecentActivities />
        </Box>

        {/* ROW 3 */}
        <Box
          gridColumn="span 8"
          gridRow="span 2"
          backgroundColor={colors.primary[400]}
        >
          <Box
            mt="8px"
            p="0 30px"
            display="flex "
            justifyContent="space-between"
            alignItems="center"
          >
            <Box>
              <Typography
                variant="h5"
                fontWeight="600"
                color={colors.grey[100]}
              >
                Campus per Department Heatmap
              </Typography>
            </Box>
            <Box>
              <IconButton component={Link} to="/app/heatmap">
                <FullScreenIcon
                  sx={{ fontSize: "26px", color: colors.greenAccent[500] }}
                />
              </IconButton>
            </Box>
          </Box>
          <Box 
          height="220px"
          padding="0px 20px">
            <CampusHeatmap isDashboard={true} />
          </Box>
        </Box>
        

        <Box
          gridColumn="span 4"
          gridRow="span 2"
          backgroundColor={colors.primary[400]}
        >
          <Typography
            variant="h5"
            fontWeight="600"
            sx={{ padding: "20px 20px 0 20px" }}
          >
            Symptom Appearance per Academic Period
          </Typography>
          <Box height="285px" mt="-20px">
            <BarChart isDashboard={true} />
          </Box>
        </Box>
      </Box>
    </Box>
  );
};

export default Dashboard;