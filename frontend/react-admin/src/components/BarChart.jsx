import { useTheme } from "@mui/material";
import { ResponsiveBar } from "@nivo/bar";
import { useState, useEffect } from "react";
import { tokens } from "../theme";

const BarChart = ({ isDashboard = false }) => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);

  // Define colors for each symptom category (matching LineChart)
  const categoryColors = {
    'Respiratory': colors.blueAccent?.[300] || '#3498db',
    'Digestive': colors.greenAccent?.[500] || '#2ecc71', 
    'Pain & Musculoskeletal': colors.redAccent?.[500] || '#e74c3c',
    'Dermatological & Trauma': colors.purpleAccent?.[300] || '#9b59b6',
    'Neurological & Psychological': colors.orangeAccent?.[500] || '#f39c12',
    'Systemic & Infectious': colors.primary?.[500] || '#f1c40f',
    'Other': colors.grey?.[100] || '#95a5a6'
  };

  // Always use shortened labels for cleaner display
  const getShortLabel = (categoryName) => {
    const shortLabels = {
      'Respiratory': 'Respiratory',
      'Digestive': 'Digestive',
      'Pain & Musculoskeletal': 'Pain/Muscle',
      'Dermatological & Trauma': 'Skin/Trauma',
      'Neurological & Psychological': 'Neuro/Psych',
      'Systemic & Infectious': 'Systemic',
      'Other': 'Other'
    };
    
    return shortLabels[categoryName] || categoryName;
  };

  useEffect(() => {
    const fetchSymptomCounts = async () => {
      try {
        setLoading(true);
        const token = localStorage.getItem('token');
        
        if (!token) {
          console.log('No token found for BarChart');
          return;
        }

        const response = await fetch(`${process.env.REACT_APP_API_URL}/api/patients/symptoms/symptom_category_counts/`, {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        });

        if (response.ok) {
          const result = await response.json();
          console.log('Symptom category data:', result);
          
          // Define preferred order for categories
          const categoryOrder = [
            'Respiratory',
            'Digestive', 
            'Pain/Muscle',
            'Skin/Trauma',
            'Neuro/Psych',
            'Systemic',
            'Other'
          ];

          // Process data with hardcoded shortened labels
          const processedData = (result.data || []).map(item => {
            let shortCategory;
            switch(item.category) {
              case 'Pain & Musculoskeletal':
                shortCategory = 'Pain/Muscle';
                break;
              case 'Dermatological & Trauma':
                shortCategory = 'Skin/Trauma';
                break;
              case 'Neurological & Psychological':
                shortCategory = 'Neuro/Psych';
                break;
              case 'Systemic & Infectious':
                shortCategory = 'Systemic';
                break;
              case 'Respiratory':
                shortCategory = 'Respiratory';
                break;
              case 'Digestive':
                shortCategory = 'Digestive';
                break;
              case 'Other':
                shortCategory = 'Other';
                break;
              default:
                shortCategory = item.category;
            }
            
            return {
              ...item,
              originalCategory: item.category,
              category: shortCategory
            };
          });

          // Sort data according to preferred order
          const sortedData = processedData.sort((a, b) => {
            const aIndex = categoryOrder.indexOf(a.category);
            const bIndex = categoryOrder.indexOf(b.category);
            
            // If category not found in order, put it at the end
            const aOrder = aIndex === -1 ? 999 : aIndex;
            const bOrder = bIndex === -1 ? 999 : bIndex;
            
            return aOrder - bOrder;
          });
          
          setData(sortedData);
        } else {
          console.error('Failed to fetch symptom category data:', response.status);
        }
      } catch (error) {
        console.error('Error fetching symptom category data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchSymptomCounts();
  }, []);

  if (loading) {
    return <div style={{ color: colors.grey[100], padding: '20px' }}>Loading symptom data...</div>;
  }

  if (!data || data.length === 0) {
    return <div style={{ color: colors.grey[100], padding: '20px' }}>No symptom data available</div>;
  }

  return (
    <ResponsiveBar
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
      }}
      keys={["count"]}
      indexBy="category"
      margin={{ top: 50, right: 130, bottom: 50, left: 60 }}
      padding={0.3}
      valueScale={{ type: "linear" }}
      indexScale={{ type: "band", round: true }}
      colors={(bar) => {
        // Find the data item that matches this bar
        const dataItem = data.find(item => item.category === bar.indexValue);
        const originalCategory = dataItem?.originalCategory || bar.indexValue;
        return categoryColors[originalCategory] || colors.grey[100];
      }}
      defs={[
        {
          id: "dots",
          type: "patternDots",
          background: "inherit",
          color: "#38bcb2",
          size: 4,
          padding: 1,
          stagger: true,
        },
        {
          id: "lines",
          type: "patternLines",
          background: "inherit",
          color: "#eed312",
          rotation: -45,
          lineWidth: 6,
          spacing: 10,
        },
      ]}
      borderColor={{
        from: "color",
        modifiers: [["darker", "1.6"]],
      }}
      axisTop={null}
      axisRight={null}
      axisBottom={null}
      axisLeft={{
        tickSize: 5,
        tickPadding: 5,
        tickRotation: 0,
        legend: isDashboard ? undefined : "Number of Cases",
        legendPosition: "middle",
        legendOffset: -40,
      }}
      enableLabel={true}
      label={(d) => `${d.value}`}
      labelSkipWidth={12}
      labelSkipHeight={12}
      labelTextColor={{
        from: "color",
        modifiers: [["darker", 1.6]],
      }}
      legends={[
        {
          dataFrom: "indexes",
          anchor: "bottom-right",
          direction: "column",
          justify: false,
          translateX: isDashboard ? 100 : 120,
          translateY: 0,
          itemsSpacing: 2,
          itemWidth: isDashboard ? 80 : 100,
          itemHeight: 20,
          itemDirection: "left-to-right",
          itemOpacity: 0.85,
          symbolSize: 20,
          effects: [
            {
              on: "hover",
              style: {
                itemOpacity: 1,
              },
            },
          ],
        },
      ]}
      role="application"
      barAriaLabel={function (e) {
        return e.indexValue + ": " + e.formattedValue + " cases";
      }}
    />
  );
};

export default BarChart;