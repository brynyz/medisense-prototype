import { Box, Typography, useTheme } from "@mui/material";
import { useState, useEffect } from "react";
import { tokens } from "../theme";
import CalendarTodayIcon from "@mui/icons-material/CalendarToday";
import AirIcon from "@mui/icons-material/Air";
import WaterDropIcon from "@mui/icons-material/WaterDrop";
import ThunderstormIcon from "@mui/icons-material/Thunderstorm";
import VisibilityIcon from "@mui/icons-material/Visibility";

const RecentActivities = () => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  const [activities, setActivities] = useState([]);
  const [loading, setLoading] = useState(true);

  // Coordinates for your campus (you can adjust these)
  const CAMPUS_LAT = 16.937232;
  const CAMPUS_LON = 121.764291;

  // Helper function to parse dates in various formats
  const parseFlexibleDate = (dateStr) => {
    if (!dateStr) return null;
    
    // Try direct parsing first
    let date = new Date(dateStr);
    if (!isNaN(date.getTime())) return date;
    
    // Try common formats if direct parsing fails
    const formats = [
      // ISO formats
      /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/, // 2024-01-15T10:30:00
      /^\d{4}-\d{2}-\d{2}/, // 2024-01-15
      
      // US formats
      /^\d{1,2}\/\d{1,2}\/\d{4}/, // 1/15/2024 or 01/15/2024
      /^\d{1,2}-\d{1,2}-\d{4}/, // 1-15-2024 or 01-15-2024
      
      // European formats
      /^\d{1,2}\/\d{1,2}\/\d{4}/, // 15/1/2024 or 15/01/2024
      /^\d{1,2}-\d{1,2}-\d{4}/, // 15-1-2024 or 15-01-2024
    ];
    
    // Try parsing with explicit format handling
    try {
      // Handle MM/DD/YYYY or DD/MM/YYYY
      if (dateStr.includes('/') || dateStr.includes('-')) {
        const separator = dateStr.includes('/') ? '/' : '-';
        const parts = dateStr.split(separator);
        
        if (parts.length >= 3) {
          // Try different arrangements
          const arrangements = [
            [parts[2], parts[0], parts[1]], // YYYY, MM, DD
            [parts[2], parts[1], parts[0]], // YYYY, DD, MM
          ];
          
          for (const [year, month, day] of arrangements) {
            const testDate = new Date(parseInt(year), parseInt(month) - 1, parseInt(day));
            if (!isNaN(testDate.getTime()) && testDate.getFullYear() > 2020) {
              return testDate;
            }
          }
        }
      }
      
      // Last resort: try to extract numbers and create a reasonable date
      const numbers = dateStr.match(/\d+/g);
      if (numbers && numbers.length >= 3) {
        // Assume the largest number is the year
        const nums = numbers.map(n => parseInt(n)).sort((a, b) => b - a);
        const year = nums.find(n => n > 2020) || new Date().getFullYear();
        const month = nums.find(n => n >= 1 && n <= 12) || 1;
        const day = nums.find(n => n >= 1 && n <= 31) || 1;
        
        return new Date(year, month - 1, day);
      }
    } catch (error) {
      console.warn('Date parsing failed for:', dateStr, error);
    }
    
    return null;
  };

  useEffect(() => {
    const fetchActivities = async () => {
      try {
        setLoading(true);
        
        // Fetch weather and AQI data from OpenMeteo
        const weatherResponse = await fetch(
          `https://api.open-meteo.com/v1/forecast?latitude=${CAMPUS_LAT}&longitude=${CAMPUS_LON}&past_days=30&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max&current=temperature_2m&timezone=Asia/Manila`
        );
        
        const aqiResponse = await fetch(
          `https://air-quality-api.open-meteo.com/v1/air-quality?latitude=${CAMPUS_LAT}&longitude=${CAMPUS_LON}&past_days=30&hourly=pm2_5,pm10,ozone&timezone=Asia/Manila`
        );

        const weatherData = await weatherResponse.json();
        const aqiData = await aqiResponse.json();

        console.log('🌤️ Weather API Response:', weatherResponse.status, weatherData);
        console.log('🌬️ AQI API Response:', aqiResponse.status, aqiData);

        // Fetch visit data for busiest month calculation
        const token = localStorage.getItem('token');
        let visitData = [];
        
        if (token) {
          try {
            const visitsResponse = await fetch(`${process.env.REACT_APP_API_URL}/api/patients/symptoms/`, {
              headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json',
              },
            });
            
            if (visitsResponse.ok) {
              const data = await visitsResponse.json();
              visitData = Array.isArray(data) ? data : (data.results || []);
            }
          } catch (error) {
            console.log('Could not fetch visit data:', error);
          }
        }

        // Process the data to create activities
        console.log('🔄 Processing activities with data:', {
          weatherDataValid: !!(weatherData && weatherData.daily),
          aqiDataValid: !!(aqiData && aqiData.hourly),
          visitDataCount: visitData.length
        });
        
        const processedActivities = await processActivitiesData(weatherData, aqiData, visitData);
        console.log('✅ Processed activities:', processedActivities);
        setActivities(processedActivities);
        
      } catch (error) {
        console.error('Error fetching activities data:', error);
        // Set fallback activities if API fails
        setActivities(getFallbackActivities());
      } finally {
        setLoading(false);
      }
    };

    fetchActivities();
  }, []);

  const processActivitiesData = async (weatherData, aqiData, visitData) => {
    const activities = [];

    try {
      // 1. Busiest Month Analysis
      if (visitData.length > 0) {
        console.log('Sample visit data for debugging:', visitData.slice(0, 3));
        
        const monthCounts = {};
        let validDates = 0;
        let invalidDates = 0;
        
        visitData.forEach((visit, index) => {
          // Try multiple date fields and formats
          let dateStr = visit.date || visit.created_at || visit.timestamp;
          if (!dateStr) {
            if (index < 5) console.log('No date found for visit:', visit);
            return;
          }
          
          // Parse date more robustly
          let date = parseFlexibleDate(dateStr);
          if (!date || isNaN(date.getTime())) {
            invalidDates++;
            if (index < 5) console.log('Failed to parse date:', dateStr, 'for visit:', visit);
            return;
          }
          
          validDates++;
          const monthKey = `${date.getFullYear()}-${date.getMonth()}`;
          const monthName = date.toLocaleDateString('en-US', { month: 'long', year: 'numeric' });
          
          if (!monthCounts[monthKey]) {
            monthCounts[monthKey] = { count: 0, name: monthName };
          }
          monthCounts[monthKey].count++;
        });

        console.log(`Date parsing results: ${validDates} valid, ${invalidDates} invalid dates`);
        console.log('Month counts:', monthCounts);

        const busiestMonth = Object.values(monthCounts).reduce((max, current) => 
          current.count > max.count ? current : max, { count: 0, name: 'No data' });

        if (busiestMonth.count > 0) {
          activities.push({
            id: 'busiest-month',
            icon: <CalendarTodayIcon sx={{ color: colors.greenAccent[500], fontSize: "20px" }} />,
            title: 'Busiest Month',
            description: `${busiestMonth.count} visits in ${busiestMonth.name}`,
            timestamp: 'This semester',
            type: 'visits'
          });
        }
      }

      // 2. Recent AQI Alert
      if (aqiData.hourly && aqiData.hourly.pm2_5) {
        const pm25Values = aqiData.hourly.pm2_5.filter(val => val !== null);
        console.log('AQI Data - PM2.5 values found:', pm25Values.length, 'Max:', Math.max(...pm25Values));
        
        if (pm25Values.length > 0) {
          const maxPM25 = Math.max(...pm25Values);
          const maxIndex = aqiData.hourly.pm2_5.indexOf(maxPM25);
          
          if (maxIndex !== -1) {
            const alertDate = new Date(aqiData.hourly.time[maxIndex]);
            const formattedDate = alertDate.toLocaleDateString('en-US', { 
              month: 'long', 
              day: 'numeric' 
            });
            
            // Lower threshold - show if PM2.5 > 25 (moderate level)
            const alertLevel = maxPM25 > 100 ? 'Critical' : maxPM25 > 50 ? 'Unhealthy' : maxPM25 > 25 ? 'Moderate' : 'Good';
            const color = maxPM25 > 100 ? colors.redAccent[500] : maxPM25 > 50 ? colors.orangeAccent[500] : colors.yellowAccent[500];
            
            activities.push({
              id: 'aqi-alert',
              icon: <AirIcon sx={{ color: color, fontSize: "20px" }} />,
              title: `Air Quality: ${alertLevel}`,
              description: `PM2.5 peaked at ${Math.round(maxPM25)} μg/m³ on ${formattedDate}`,
              timestamp: getRelativeTime(alertDate),
              type: 'environmental'
            });
          }
        }
      } else {
        console.log('No AQI data available');
      }

      // 3. Weather Events - Highest Rainfall
      if (weatherData.daily && weatherData.daily.precipitation_sum) {
        const precipValues = weatherData.daily.precipitation_sum.filter(val => val !== null && val > 0);
        console.log('Weather Data - Precipitation values:', precipValues.length, 'Max:', precipValues.length > 0 ? Math.max(...precipValues) : 'None');
        
        if (precipValues.length > 0) {
          const maxPrecip = Math.max(...precipValues);
          const maxIndex = weatherData.daily.precipitation_sum.indexOf(maxPrecip);
          
          // Show highest rainfall if significant (> 5mm)
          if (maxIndex !== -1 && maxPrecip > 5) {
            const rainDate = new Date(weatherData.daily.time[maxIndex]);
            const dayName = rainDate.toLocaleDateString('en-US', { weekday: 'long' });
            
            // More detailed intensity classification
            let intensity, color;
            if (maxPrecip > 50) {
              intensity = 'Extreme';
              color = colors.redAccent[500];
            } else if (maxPrecip > 25) {
              intensity = 'Heavy';
              color = colors.orangeAccent[500];
            } else if (maxPrecip > 10) {
              intensity = 'Moderate';
              color = colors.blueAccent[500];
            } else {
              intensity = 'Light';
              color = colors.blueAccent[300];
            }
            
            activities.push({
              id: 'highest-rainfall',
              icon: <WaterDropIcon sx={{ color: color, fontSize: "20px" }} />,
              title: 'Highest Rainfall',
              description: `${intensity}: ${maxPrecip.toFixed(1)}mm last ${dayName}`,
              timestamp: getRelativeTime(rainDate),
              type: 'weather'
            });
          }
        }
      } else {
        console.log('No precipitation data available');
      }

      // 4. Temperature Extreme
      if (weatherData.daily && weatherData.daily.temperature_2m_max) {
        const tempValues = weatherData.daily.temperature_2m_max.filter(val => val !== null);
        if (tempValues.length > 0) {
          const maxTemp = Math.max(...tempValues);
          const maxIndex = weatherData.daily.temperature_2m_max.indexOf(maxTemp);
          
          if (maxIndex !== -1) {
            const tempDate = new Date(weatherData.daily.time[maxIndex]);
            const dayName = tempDate.toLocaleDateString('en-US', { weekday: 'long' });
            
            activities.push({
              id: 'temperature-event',
              icon: <ThunderstormIcon sx={{ color: colors.orangeAccent[500], fontSize: "20px" }} />,
              title: 'Temperature Peak',
              description: `Highest temperature: ${maxTemp.toFixed(1)}°C last ${dayName}`,
              timestamp: getRelativeTime(tempDate),
              type: 'weather'
            });
          }
        }
      }

      // 5. Extreme Weather Events Detection
      if (weatherData.daily) {
        const extremeEvents = [];
        
        // Heat Wave Detection (3+ consecutive days > 35°C)
        if (weatherData.daily.temperature_2m_max) {
          const temps = weatherData.daily.temperature_2m_max;
          let consecutiveHotDays = 0;
          let heatWaveStart = -1;
          
          for (let i = 0; i < temps.length; i++) {
            if (temps[i] > 35) {
              if (consecutiveHotDays === 0) heatWaveStart = i;
              consecutiveHotDays++;
            } else {
              if (consecutiveHotDays >= 3) {
                const startDate = new Date(weatherData.daily.time[heatWaveStart]);
                const endDate = new Date(weatherData.daily.time[i - 1]);
                extremeEvents.push({
                  type: 'heatwave',
                  severity: 'extreme',
                  startDate,
                  endDate,
                  duration: consecutiveHotDays,
                  maxTemp: Math.max(...temps.slice(heatWaveStart, i))
                });
              }
              consecutiveHotDays = 0;
            }
          }
          
          // Check if heat wave extends to the end
          if (consecutiveHotDays >= 3) {
            const startDate = new Date(weatherData.daily.time[heatWaveStart]);
            const endDate = new Date(weatherData.daily.time[temps.length - 1]);
            extremeEvents.push({
              type: 'heatwave',
              severity: 'extreme',
              startDate,
              endDate,
              duration: consecutiveHotDays,
              maxTemp: Math.max(...temps.slice(heatWaveStart))
            });
          }
        }
        
        // Extreme Rainfall Detection (>50mm in a day or >100mm in 3 days)
        if (weatherData.daily.precipitation_sum) {
          const precip = weatherData.daily.precipitation_sum;
          
          // Single day extreme rainfall
          for (let i = 0; i < precip.length; i++) {
            if (precip[i] > 40) {
              const date = new Date(weatherData.daily.time[i]);
              extremeEvents.push({
                type: 'extreme_rain',
                severity: precip[i] > 100 ? 'severe' : 'extreme',
                date,
                amount: precip[i]
              });
            }
          }
          
          // Multi-day extreme rainfall
          for (let i = 0; i < precip.length - 2; i++) {
            const threeDayTotal = precip[i] + precip[i + 1] + precip[i + 2];
            if (threeDayTotal > 100) {
              const startDate = new Date(weatherData.daily.time[i]);
              const endDate = new Date(weatherData.daily.time[i + 2]);
              extremeEvents.push({
                type: 'prolonged_rain',
                severity: 'extreme',
                startDate,
                endDate,
                amount: threeDayTotal
              });
            }
          }
        }
        
        // High Wind Events (>60 km/h)
        if (weatherData.daily.windspeed_10m_max) {
          const winds = weatherData.daily.windspeed_10m_max;
          for (let i = 0; i < winds.length; i++) {
            if (winds[i] > 60) {
              const date = new Date(weatherData.daily.time[i]);
              extremeEvents.push({
                type: 'high_wind',
                severity: winds[i] > 80 ? 'severe' : 'extreme',
                date,
                windSpeed: winds[i]
              });
            }
          }
        }
        
        console.log('Extreme weather events detected:', extremeEvents);
        
        // Add the most recent/severe extreme event
        if (extremeEvents.length > 0) {
          // Sort by severity and recency
          const sortedEvents = extremeEvents.sort((a, b) => {
            const severityScore = { severe: 3, extreme: 2, moderate: 1 };
            const aSeverity = severityScore[a.severity] || 0;
            const bSeverity = severityScore[b.severity] || 0;
            
            if (aSeverity !== bSeverity) return bSeverity - aSeverity;
            
            const aDate = a.date || a.endDate || a.startDate;
            const bDate = b.date || b.endDate || b.startDate;
            return new Date(bDate) - new Date(aDate);
          });
          
          const event = sortedEvents[0];
          let icon, title, description, color;
          
          switch (event.type) {
            case 'heatwave':
              icon = <ThunderstormIcon sx={{ color: colors.redAccent[500], fontSize: "20px" }} />;
              title = 'Heat Wave Alert';
              description = `${event.duration} days above 35°C (peak: ${event.maxTemp.toFixed(1)}°C)`;
              color = colors.redAccent[500];
              break;
              
            case 'extreme_rain':
              icon = <WaterDropIcon sx={{ color: colors.redAccent[500], fontSize: "20px" }} />;
              title = 'Extreme Rainfall';
              description = `${event.amount.toFixed(1)}mm in a single day`;
              color = colors.redAccent[500];
              break;
              
            case 'prolonged_rain':
              icon = <WaterDropIcon sx={{ color: colors.orangeAccent[500], fontSize: "20px" }} />;
              title = 'Prolonged Heavy Rain';
              description = `${event.amount.toFixed(1)}mm over 3 days`;
              color = colors.orangeAccent[500];
              break;
              
            case 'high_wind':
              icon = <AirIcon sx={{ color: colors.orangeAccent[500], fontSize: "20px" }} />;
              title = 'High Wind Event';
              description = `Wind speeds reached ${event.windSpeed.toFixed(1)} km/h`;
              color = colors.orangeAccent[500];
              break;
          }
          
          const eventDate = event.date || event.endDate || event.startDate;
          
          activities.push({
            id: 'extreme-weather',
            icon,
            title,
            description,
            timestamp: getRelativeTime(eventDate),
            type: 'extreme_weather'
          });
        }
      }

    } catch (error) {
      console.error('Error processing activities data:', error);
    }

    console.log('Final activities before sorting:', activities);

    // If we have very few activities, add some informational ones
    if (activities.length < 2) {
      // Add a general weather status using current temperature
      if (weatherData.current && weatherData.current.temperature_2m) {
        const currentTemp = weatherData.current.temperature_2m;
        activities.push({
          id: 'current-weather',
          icon: <ThunderstormIcon sx={{ color: colors.greenAccent[500], fontSize: "20px" }} />,
          title: 'Current Weather',
          description: `Temperature: ${currentTemp.toFixed(1)}°C now`,
          timestamp: 'Current',
          type: 'weather'
        });
      }

      // Add a general AQI status
      if (aqiData.hourly && aqiData.hourly.pm2_5) {
        const recentPM25 = aqiData.hourly.pm2_5[aqiData.hourly.pm2_5.length - 1];
        if (recentPM25) {
          activities.push({
            id: 'current-aqi',
            icon: <AirIcon sx={{ color: colors.blueAccent[500], fontSize: "20px" }} />,
            title: 'Air Quality',
            description: `PM2.5: ${Math.round(recentPM25)} μg/m³ currently`,
            timestamp: 'Current',
            type: 'environmental'
          });
        }
      }
    }

    // Sort by recency and return top 4
    return activities
      .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
      .slice(0, 4);
  };

  const getFallbackActivities = () => [
    {
      id: 'fallback-1',
      icon: <CalendarTodayIcon sx={{ color: colors.greenAccent[500], fontSize: "20px" }} />,
      title: 'Busiest Month',
      description: 'Loading visit data...',
      timestamp: 'This semester',
      type: 'visits'
    },
    {
      id: 'fallback-2',
      icon: <AirIcon sx={{ color: colors.redAccent[500], fontSize: "20px" }} />,
      title: 'AQI Status',
      description: 'Loading air quality data...',
      timestamp: 'Recent',
      type: 'environmental'
    },
    {
      id: 'fallback-3',
      icon: <WaterDropIcon sx={{ color: colors.blueAccent[500], fontSize: "20px" }} />,
      title: 'Weather Update',
      description: 'Loading weather data...',
      timestamp: 'Recent',
      type: 'weather'
    }
  ];

  const getRelativeTime = (date) => {
    const now = new Date();
    const diffTime = Math.abs(now - date);
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;
    if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;
    return `${Math.floor(diffDays / 30)} months ago`;
  };

  const getActivityColor = (type) => {
    switch (type) {
      case 'visits': return colors.greenAccent[500];
      case 'environmental': return colors.redAccent[500];
      case 'weather': return colors.blueAccent[500];
      case 'extreme_weather': return colors.redAccent[500];
      default: return colors.grey[100];
    }
  };

  console.log('🎨 Rendering RecentActivities:', { loading, activitiesCount: activities.length, activities });

  if (loading) {
    return (
      <Box>
        <Box
          display="flex"
          justifyContent="space-between"
          alignItems="center"
          borderBottom={`4px solid ${colors.primary[500]}`}
          colors={colors.grey[100]}
          p="15px"
        >
          <Typography color={colors.grey[100]} variant="h5" fontWeight="600">
            Recent Activities
          </Typography>
        </Box>
        <Box p="15px">
          <Typography color={colors.grey[100]}>Loading activities...</Typography>
        </Box>
      </Box>
    );
  }

  return (
    <Box>
      <Box
        display="flex"
        justifyContent="space-between"
        alignItems="center"
        borderBottom={`4px solid ${colors.primary[500]}`}
        colors={colors.grey[100]}
        p="15px"
      >
        <Typography color={colors.grey[100]} variant="h5" fontWeight="600">
          Recent Activities
        </Typography>
      </Box>
      
      {activities.map((activity) => (
        <Box
          key={activity.id}
          display="flex"
          justifyContent="space-between"
          alignItems="center"
          borderBottom={`4px solid ${colors.primary[500]}`}
          p="15px"
        >
          <Box display="flex" alignItems="center" gap="12px">
            {activity.icon}
            <Box>
              <Typography
                color={getActivityColor(activity.type)}
                variant="h6"
                fontWeight="600"
              >
                {activity.title}
              </Typography>
              <Typography color={colors.grey[100]} variant="body2">
                {activity.description}
              </Typography>
            </Box>
          </Box>
          <Box>
            <Typography color={colors.grey[300]} variant="body2">
              {activity.timestamp}
            </Typography>
          </Box>
        </Box>
      ))}
      
      {activities.length === 0 && (
        <Box p="15px">
          <Typography color={colors.grey[100]}>No recent activities available</Typography>
        </Box>
      )}
    </Box>
  );
};

export default RecentActivities;
