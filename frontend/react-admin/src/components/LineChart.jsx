import { ResponsiveLine } from "@nivo/line";
import { useTheme, Box, Typography } from "@mui/material";
import { useState, useEffect } from "react";
import { tokens } from "../theme";

const LineChart = ({ isCustomLineColors = false, isDashboard = false }) => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Define colors for each symptom category (matching your ML pipeline categories)
  const categoryColors = {
    'Respiratory': colors.blueAccent?.[300] || '#3498db',
    'Digestive': colors.greenAccent?.[500] || '#2ecc71', 
    'Pain & Musculoskeletal': colors.redAccent?.[500] || '#e74c3c',
    'Dermatological & Trauma': colors.purpleAccent?.[300] || '#9b59b6',
    'Neurological & Psychological': colors.orangeAccent?.[500] || '#f39c12',
    'Systemic & Infectious': colors.primary?.[500] || '#f1c40f',
    'Other': colors.grey?.[100] || '#95a5a6'
  };

  // Shortened labels for dashboard display
  const getLabelForDashboard = (categoryId) => {
    if (!isDashboard) return categoryId;
    
    const shortLabels = {
      'Respiratory': 'Respiratory',
      'Digestive': 'Digestive',
      'Pain & Musculoskeletal': 'Pain/Muscle',
      'Dermatological & Trauma': 'Skin/Trauma',
      'Neurological & Psychological': 'Neuro/Psych',
      'Systemic & Infectious': 'Systemic',
      'Other': 'Other'
    };
    
    return shortLabels[categoryId] || categoryId;
  };

  // Fetch real symptom trends data
  useEffect(() => {
    const fetchSymptomTrends = async () => {
      try {
        setLoading(true);
        setError(null);
        
        const token = localStorage.getItem('token');
        if (!token) {
          throw new Error('No authentication token found');
        }

        const apiUrl = `${process.env.REACT_APP_API_URL}/api/patients/symptoms/symptom_trends/`;
        console.log('🔗 Attempting API call to:', apiUrl);
        console.log('🔑 Using token:', token ? `${token.substring(0, 20)}...` : 'No token');
        
        const response = await fetch(apiUrl, {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
        });

        console.log('📡 Response received:', {
          status: response.status,
          statusText: response.statusText,
          headers: Object.fromEntries(response.headers.entries())
        });

        if (!response.ok) {
          const errorText = await response.text();
          console.error('❌ API Error Details:', {
            status: response.status,
            statusText: response.statusText,
            body: errorText
          });
          throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
        }

        const result = await response.json();
        console.log('✅ Symptom trends API success:', result);
        
        // Add colors and shortened labels to the data
        const dataWithColors = result.data.map(category => ({
          ...category,
          id: getLabelForDashboard(category.id),
          color: categoryColors[category.id] || colors.grey[500]
        }));
        
        console.log('📊 Using real API data, not fallback');
        setData(dataWithColors);
      } catch (error) {
        console.error('Error fetching symptom trends:', error);
        setError(error.message);
        
        // Fallback to mock data if API fails
        console.log('⚠️ API failed, using fallback data from Sep 2022');
        setData(generateFallbackData());
      } finally {
        setLoading(false);
      }
    };

    fetchSymptomTrends();
  }, []);

  // Fallback mock data generator
  const generateFallbackData = () => {
    // Use a realistic date range matching your actual data (Sep 2022 to current)
    const startDate = new Date('2022-09-01'); // September 2022
    const endDate = new Date(); // Current date
    
    const categories = Object.keys(categoryColors);
    
    return categories.map(categoryId => {
      const categoryData = [];
      const currentDate = new Date(startDate);
      
      while (currentDate <= endDate) {
        const visits = Math.floor(Math.random() * 15) + 1; // 1-15 visits
        
        categoryData.push({
          x: currentDate.toISOString().split('T')[0],
          y: visits
        });
        
        // Move to next week
        currentDate.setDate(currentDate.getDate() + 7);
      }
      
      return {
        id: getLabelForDashboard(categoryId),
        color: categoryColors[categoryId],
        data: categoryData
      };
    });
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="400px">
        <Typography>Loading symptom trends...</Typography>
      </Box>
    );
  }

  if (error && data.length === 0) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="400px">
        <Typography color="error">Error loading data: {error}</Typography>
      </Box>
    );
  }

  return (
    <ResponsiveLine
      data={data}
      theme={{
        axis: {
          domain: {
            line: {
              stroke: colors.grey[100],
            },
          },
          legend: {
            text: {
              fill: colors.grey[100],
            },
          },
          ticks: {
            line: {
              stroke: colors.grey[100],
              strokeWidth: 1,
            },
            text: {
              fill: colors.grey[100],
            },
          },
        },
        legends: {
          text: {
            fill: colors.grey[100],
          },
        },
        tooltip: {
          container: {
            color: colors.primary[500],
          },
        },
      }}
      colors={isDashboard ? { datum: "color" } : { scheme: "nivo" }}
      margin={{ top: 50, right: 110, bottom: 70, left: 60 }}
      xScale={{ 
        type: "time",
        format: "%Y-%m-%d",
        useUTC: false,
        precision: "day"
      }}
      xFormat="time:%Y-%m-%d"
      yScale={{
        type: "linear",
        min: "auto",
        max: "auto",
        stacked: false,
        reverse: false,
      }}
      yFormat=" >-.0f"
      curve="catmullRom"
      axisTop={null}
      axisRight={null}
      axisBottom={{
        orient: "bottom",
        tickSize: 5,
        tickPadding: 5,
        tickRotation: isDashboard ? -45 : -45,
        format: isDashboard ? "%b %Y" : "%b %Y",
        tickValues: isDashboard ? "every 6 months" : "every 3 months",
        legend: isDashboard ? undefined : "Date",
        legendOffset: 50,
        legendPosition: "middle",
      }}
      axisLeft={{
        orient: "left",
        tickValues: 5,
        tickSize: 5,
        tickPadding: 5,
        tickRotation: 0,
        legend: isDashboard ? undefined : "Clinic Visits",
        legendOffset: -40,
        legendPosition: "middle",
      }}
      enableGridX={false}
      enableGridY={false}
      pointSize={8}
      pointColor={{ theme: "background" }}
      pointBorderWidth={2}
      pointBorderColor={{ from: "serieColor" }}
      pointLabelYOffset={-12}
      useMesh={true}
      enableSlices="x"
      enableCrosshair={!isDashboard}
      crosshairType="cross"
      legends={isDashboard ? [
        {
          anchor: "bottom-right",
          direction: "column",
          justify: false,
          translateX: 90,
          translateY: 0,
          itemsSpacing: 2,
          itemDirection: "left-to-right",
          itemWidth: 85,
          itemHeight: 16,
          itemOpacity: 0.75,
          symbolSize: 10,
          symbolShape: "circle",
          symbolBorderColor: "rgba(0, 0, 0, .5)",
          toggleSerie: true,
          itemTextColor: colors.grey[100],
          effects: [
            {
              on: "hover",
              style: {
                itemBackground: "rgba(0, 0, 0, .03)",
                itemOpacity: 1,
              },
            },
          ],
        },
      ] : [
        {
          anchor: "bottom-right",
          direction: "column",
          justify: false,
          translateX: 100,
          translateY: 0,
          itemsSpacing: 0,
          itemDirection: "left-to-right",
          itemWidth: 120,
          itemHeight: 20,
          itemOpacity: 0.75,
          symbolSize: 12,
          symbolShape: "circle",
          symbolBorderColor: "rgba(0, 0, 0, .5)",
          toggleSerie: true,
          effects: [
            {
              on: "hover",
              style: {
                itemBackground: "rgba(0, 0, 0, .03)",
                itemOpacity: 1,
              },
            },
          ],
        },
      ]}
    />
  );
};

export default LineChart;