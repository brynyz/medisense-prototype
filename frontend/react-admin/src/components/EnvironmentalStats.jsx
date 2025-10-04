import { Box, Typography, useTheme } from "@mui/material";
import { useState, useEffect } from "react";
import { tokens } from "../theme";
import ThermostatIcon from "@mui/icons-material/Thermostat";
import OpacityIcon from "@mui/icons-material/Opacity";
import CloudIcon from "@mui/icons-material/Cloud";
import AirIcon from "@mui/icons-material/Air";

const EnvironmentalStats = () => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  const [envData, setEnvData] = useState({
    temperature: 28.5,
    humidity: 72,
    rainfall: 2.3,
    aqi: 45
  });
  const [loading, setLoading] = useState(false);

  // Fetch real environmental data from Open-Meteo API
  useEffect(() => {
    const fetchEnvironmentalData = async () => {
      try {
        setLoading(true);
        
        // Coordinates
        const latitude =  16.937232;
        const longitude = 121.764291;
        
        // Fetch current weather data
        const weatherResponse = await fetch(
          `https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current=temperature_2m,relative_humidity_2m,precipitation&timezone=Asia/Manila`
        );
        
        // Fetch air quality data
        const airQualityResponse = await fetch(
          `https://air-quality-api.open-meteo.com/v1/air-quality?latitude=${latitude}&longitude=${longitude}&current=pm2_5,pm10,european_aqi&timezone=Asia/Manila`
        );
        
        if (weatherResponse.ok && airQualityResponse.ok) {
          const weatherData = await weatherResponse.json();
          const airQualityData = await airQualityResponse.json();
          
          const newEnvData = {
            temperature: Math.round(weatherData.current.temperature_2m * 10) / 10,
            humidity: Math.round(weatherData.current.relative_humidity_2m),
            rainfall: Math.round(weatherData.current.precipitation * 10) / 10,
            aqi: Math.round(airQualityData.current.european_aqi || 0)
          };
          
          setEnvData(newEnvData);
          console.log('✅ Environmental data updated:', newEnvData);
        } else {
          throw new Error('Failed to fetch environmental data');
        }
      } catch (error) {
        // console.error('Error fetching environmental data:', error);
        
        // // Fallback to mock data if API fails
        // const fallbackData = {
        //   temperature: Math.round((25 + Math.random() * 10) * 10) / 10,
        //   humidity: Math.round(60 + Math.random() * 30),
        //   rainfall: Math.round(Math.random() * 5 * 10) / 10,
        //   aqi: Math.round(30 + Math.random() * 50)
        // };
        // setEnvData(fallbackData);
        // console.log('⚠️ Using fallback data due to API error');
      } finally {
        setLoading(false);
      }
    };

    fetchEnvironmentalData();
    
    // Update every 5 minutes
    const interval = setInterval(fetchEnvironmentalData, 300000);
    
    return () => clearInterval(interval);
  }, []);

  const getAQIColor = (aqi) => {
    if (aqi <= 50) return colors.greenAccent[500]; // Good
    if (aqi <= 100) return colors.blueAccent[400]; // Moderate (using blue as yellow substitute)
    if (aqi <= 150) return colors.redAccent[300]; // Unhealthy for sensitive (lighter red)
    return colors.redAccent[500]; // Unhealthy
  };

  const getAQIStatus = (aqi) => {
    if (aqi <= 50) return "Good";
    if (aqi <= 100) return "Moderate";
    if (aqi <= 150) return "Unhealthy for Sensitive";
    return "Unhealthy";
  };

  return (
    <Box
      display="grid"
      gridTemplateColumns="1fr 1fr 1fr 1fr"
      gap="6px"
      height="100%"
      width="100%"
      p="6px"
    >
      {/* Temperature */}
      <Box
        display="flex"
        flexDirection="column"
        alignItems="center"
        justifyContent="center"
        backgroundColor={colors.primary[500]}
        borderRadius="6px"
        p="6px"
      >
        <ThermostatIcon sx={{ color: colors.redAccent[500], fontSize: "18px", mb: "2px" }} />
        <Typography variant="body2" fontWeight="bold" color={colors.grey[100]} fontSize="12px">
          {loading ? "..." : `${envData.temperature}°C`}
        </Typography>
        <Typography variant="caption" color={colors.grey[300]} fontSize="9px">
          Temp
        </Typography>
      </Box>

      {/* Humidity */}
      <Box
        display="flex"
        flexDirection="column"
        alignItems="center"
        justifyContent="center"
        backgroundColor={colors.primary[500]}
        borderRadius="6px"
        p="6px"
      >
        <OpacityIcon sx={{ color: colors.blueAccent[500], fontSize: "18px", mb: "2px" }} />
        <Typography variant="body2" fontWeight="bold" color={colors.grey[100]} fontSize="12px">
          {loading ? "..." : `${envData.humidity}%`}
        </Typography>
        <Typography variant="caption" color={colors.grey[300]} fontSize="9px">
          Humidity
        </Typography>
      </Box>

      {/* Rainfall */}
      <Box
        display="flex"
        flexDirection="column"
        alignItems="center"
        justifyContent="center"
        backgroundColor={colors.primary[500]}
        borderRadius="6px"
        p="6px"
      >
        <CloudIcon sx={{ color: colors.grey[400], fontSize: "18px", mb: "2px" }} />
        <Typography variant="body2" fontWeight="bold" color={colors.grey[100]} fontSize="12px">
          {loading ? "..." : `${envData.rainfall}mm`}
        </Typography>
        <Typography variant="caption" color={colors.grey[300]} fontSize="9px">
          Rain
        </Typography>
      </Box>

      {/* AQI */}
      <Box
        display="flex"
        flexDirection="column"
        alignItems="center"
        justifyContent="center"
        backgroundColor={colors.primary[500]}
        borderRadius="6px"
        p="6px"
      >
        <AirIcon sx={{ color: getAQIColor(envData.aqi), fontSize: "18px", mb: "2px" }} />
        <Typography variant="body2" fontWeight="bold" color={colors.grey[100]} fontSize="12px">
          {loading ? "..." : envData.aqi}
        </Typography>
        <Typography variant="caption" color={colors.grey[300]} fontSize="9px">
          AQI
        </Typography>
      </Box>
    </Box>
  );
};

export default EnvironmentalStats;
