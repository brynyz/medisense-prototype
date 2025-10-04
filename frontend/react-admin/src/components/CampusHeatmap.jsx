import React, { useRef, useEffect, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import { geoFeatures } from "../data/isuGeoFeatures";

mapboxgl.accessToken = 'pk.eyJ1IjoiYnJ5bGxlMDUiLCJhIjoiY21nYjRyODRzMGV2ODJrc2FldWZqdzk4OCJ9.o_C8hQDfqdIsXvsd9oeW-w';

const CampusHeatmap = ({ isDashboard = false }) => {

  const mapContainer = useRef(null);
  const map = useRef(null);
  const [status, setStatus] = useState('Initializing...');
  const [logs, setLogs] = useState([]);
  const [departmentData, setDepartmentData] = useState(null);
  const [loading, setLoading] = useState(true);

  const addLog = (message) => {
    console.log(message);
    setLogs(prev => [...prev, `${new Date().toLocaleTimeString()}: ${message}`]);
  };

  // Fetch department visit data from API
  const fetchDepartmentData = async () => {
    try {
      setLoading(true);
      const token = localStorage.getItem('token');
      console.log('🔑 Token:', token ? 'Found' : 'Missing');
      
      // Test base endpoint first
      console.log('🧪 Testing base symptoms endpoint...');
      const testResponse = await fetch('http://localhost:8000/api/patients/symptoms/', {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });
      console.log('🧪 Base symptoms endpoint status:', testResponse.status);
      
      const response = await fetch('http://localhost:8000/api/patients/symptoms/department_visits/', {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });

      console.log('📡 API Response status:', response.status);
      
      if (response.ok) {
        const data = await response.json();
        console.log('✅ Department data fetched:', data);
        setDepartmentData(data);
      } else {
        const errorText = await response.text();
        console.error('❌ Failed to fetch department data:', response.status, errorText);
        // Use fallback mock data if API fails
        setDepartmentData({
          'ccsict': 45,
          'ccje': 32,
          'cbm': 28,
          'ced': 35,
          'agri': 22,
          'poly': 18,
          'sas': 25
        });
      }
    } catch (error) {
      console.error('💥 Error fetching department data:', error);
      // Use fallback mock data if API fails (uppercase to match API format)
      setDepartmentData({
        'CCSICT': 45,
        'CCJE': 32,
        'CBM': 28,
        'CED': 35,
        'AGRI': 22,
        'POLY': 18,
        'SAS': 25,
        'LAW': 11,
        'STAFF': 12,
        'OTH': 10,
        'GRAD': 0
      });
    } finally {
      setLoading(false);
    }
  };

  // Fetch data on component mount
  useEffect(() => {
    console.log('🚀 CampusHeatmap component mounted, starting data fetch...');
    
    // TEMPORARY: Force immediate data for testing
    console.log('🧪 Setting test data immediately...');
    setDepartmentData({
      'CCSICT': 23,
      'CCJE': 26,
      'CBM': 22,
      'CED': 35,
      'AGRI': 7,
      'POLY': 2,
      'SAS': 7,
      'LAW': 11,
      'STAFF': 12,
      'OTH': 10,
      'GRAD': 0
    });
    setLoading(false);
    
    // Also try the API call
    fetchDepartmentData();
    
    // Fallback: if data doesn't load within 10 seconds, use mock data
    const fallbackTimer = setTimeout(() => {
      if (!departmentData && !loading) {
        console.log('⏰ API timeout - using fallback data');
        setDepartmentData({
          'CCSICT': 23,
          'CCJE': 26,
          'CBM': 22,
          'CED': 35,
          'AGRI': 7,
          'POLY': 2,
          'SAS': 7,
          'LAW': 11,
          'STAFF': 12,
          'OTH': 10,
          'GRAD': 0
        });
      }
    }, 10000);
    
    return () => clearTimeout(fallbackTimer);
  }, []);

  useEffect(() => {
    if (map.current) return;

    // addLog('🔧 Starting campus map...');
    // addLog(`🔑 Token: ${mapboxgl.accessToken.substring(0, 20)}...`);
    
    // setStatus('Testing network connectivity...');

    // Test network connectivity first
    fetch(`https://api.mapbox.com/styles/v1/mapbox/streets-v11?access_token=${mapboxgl.accessToken}`)
      .then(response => {
        // addLog(`🌐 Network test response: ${response.status}`);
        if (response.ok) {
          // addLog('✅ Network and token valid, creating map...');
          // setStatus('Network OK, creating map...');
          createMap();
        } else {
          // addLog('❌ Token/network validation failed');
          // setStatus(`❌ Token invalid (${response.status})`);
        }
      })
      .catch(error => {
        // addLog(`❌ Network error: ${error.message}`);
        // setStatus(`❌ Network error: ${error.message}`);
        // Try creating map anyway in case it's just a CORS issue
        // addLog('🔄 Trying to create map anyway...');
        createMap();
      });

    function createMap() {
      try {
        // addLog('🗺️ Creating Mapbox map instance...');
        
        map.current = new mapboxgl.Map({
          container: mapContainer.current,
          style: 'mapbox://styles/mapbox/standard',
          config: {
            basemap: {
              lightPreset: "night",
              showPlaceLabels: false,
              showRoadLabels: false,
              showTransitLabels: false,
              showAdminBoundaries: false,
              show3dObjects: isDashboard ? false : true,
              showPointOfInterestLabels: false
            }
          },
          center: [121.7645, 16.938], // Default to campus location
          zoom: isDashboard ? 16.5 : 18,
          bearing: -78, // Default rotation
          attributionControl: false,
          pitch: isDashboard ? 0 : 50,
          // Lock map interactions
          // interactive: !isDashboard, // Disable all interactions for dashboard
          scrollZoom: !isDashboard,
          boxZoom: false,
          dragRotate: false,
          // dragPan: !isDashboard,
          // keyboard: false,
          doubleClickZoom: false,
          // touchZoomRotate: false
        });

    //   addLog('✅ Map instance created, waiting for events...');
    //   setStatus('Map created, waiting for load...');

      const timeout = setTimeout(() => {
        // addLog('⏰ Map load timeout after 8 seconds');
        // setStatus('❌ TIMEOUT: Map failed to load within 8 seconds');
      }, 8000);

      map.current.on('load', () => {
        clearTimeout(timeout);
        // addLog('🎉 Map loaded successfully!');
        // setStatus('✅ SUCCESS: Map loaded!');
        
        // Don't add features here - wait for data to load via useEffect
      });

      map.current.on('error', (e) => {
        clearTimeout(timeout);
        // addLog(`❌ Map error: ${JSON.stringify(e.error)}`);
        // setStatus(`❌ Map error: ${e.error?.message || 'Unknown'}`);
      });

      } catch (err) {
        // addLog(`💥 Exception creating map: ${err.message}`);
        // setStatus(`❌ Exception: ${err.message}`);
      }
    }

    return () => {
      if (map.current) {
        // addLog('🧹 Cleaning up map...');
        map.current.remove();
      }
    };
  }, []);

  // Add resize handler for dynamic width when sidebar toggles
  useEffect(() => {
    const handleResize = () => {
      if (map.current) {
        // Trigger map resize after a short delay to allow CSS transitions to complete
        setTimeout(() => {
          map.current.resize();
        }, 300);
      }
    };

    // Listen for window resize events
    window.addEventListener('resize', handleResize);
    
    // Also listen for sidebar toggle by observing the content element class changes
    const contentElement = document.querySelector('.content');
    if (contentElement) {
      const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
          if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
            handleResize();
          }
        });
      });
      
      observer.observe(contentElement, {
        attributes: true,
        attributeFilter: ['class']
      });

      return () => {
        window.removeEventListener('resize', handleResize);
        observer.disconnect();
      };
    }
  }, []);

  // Add campus features when data is loaded and map is ready
  useEffect(() => {
    if (!departmentData || loading || !map.current) return;
    
    // Try to add features with a small delay to ensure map is ready
    const addFeatures = () => {
      try {
        // Ensure map is fully loaded and methods are available
        if (!map.current || !map.current.getSource || !map.current.isStyleLoaded()) {
          setTimeout(addFeatures, 100);
          return;
        }
        
        // Clear existing source and layers first
        if (map.current.getSource('buildings')) {
          if (map.current.getLayer('buildings-fill')) {
            map.current.removeLayer('buildings-fill');
          }
          if (map.current.getLayer('buildings-stroke')) {
            map.current.removeLayer('buildings-stroke');
          }
          if (map.current.getLayer('buildings-labels')) {
            map.current.removeLayer('buildings-labels');
          }
          map.current.removeSource('buildings');
        }
        addCampusFeatures(departmentData);
      } catch (error) {
        // If map isn't ready, try again after a short delay
        setTimeout(addFeatures, 100);
      }
    };
    
    // Check if map is loaded, if not wait for it
    if (map.current.isStyleLoaded && map.current.isStyleLoaded()) {
      addFeatures();
    } else {
      map.current.once('styledata', addFeatures);
    }
  }, [departmentData, loading]);

  const addCampusFeatures = (dataOverride = null) => {
    try {
      const dataToUse = dataOverride || departmentData;
      
      if (!dataToUse) {
        return;
      }
      
      // Add buildings directly (no flying animation needed)
        const buildings = geoFeatures.features.filter(
          feature => feature.properties.building_name && !feature.properties.general_area
        );
        
        const buildingNames = buildings.map(b => b.properties.building_name);

        // Define which buildings should show clinic visit data
        const departmentBuildings = [
          'CCSICT', 'CCJE', 'CBM', 'EDUC', 'Agriculture', 'Polytechnic', 'SAS',
          'LAW', 'Administration_Building', 'Commercial Building', 'Graduate_School'
        ];
        const nonDepartmentBuildings = ['Clinic', 'Food_Court'];
        

        // Use real department visit data from API (uppercase keys from Django)
        const departmentVisits = dataToUse || {
          'CCSICT': 0,
          'CCJE': 0,
          'CBM': 0,
          'CED': 0,
          'AGRI': 0,
          'POLY': 0,
          'SAS': 0,
          'LAW': 0,
          'STAFF': 0,
          'OTH': 0,
          'GRAD': 0
        };

        // Map API department codes (UPPERCASE from API) to building names (as they appear in geoFeatures)
        const departmentMapping = {
          'CCSICT': 'CCSICT',
          'CCJE': 'CCJE', 
          'CBM': 'CBM',
          'CED': 'EDUC',        // Education building
          'AGRI': 'Agriculture',
          'POLY': 'Polytechnic',
          'SAS': 'SAS',
          'LAW': 'LAW',         // Law building
          'STAFF': 'Administration_Building',  // Staff goes to admin
          'OTH': 'Commercial Building',        // Others go to commercial (note: space not underscore)
          'GRAD': 'Graduate_School'            // Graduate programs
        };

        // Convert API data to building names
        const buildingVisits = {};
        Object.entries(departmentVisits).forEach(([apiCode, count]) => {
          const buildingName = departmentMapping[apiCode.toUpperCase()];
          if (buildingName) {
            buildingVisits[buildingName] = count;
          }
        });
        

        if (buildings.length > 0) {
          // Add visit counts - only for department buildings
          const buildingsWithData = buildings.map(feature => {
            const buildingName = feature.properties.building_name;
            const isDepartmentBuilding = departmentBuildings.includes(buildingName);
            const visitCount = isDepartmentBuilding ? buildingVisits[buildingName] || 0 : 0;
            
            
            return {
              ...feature,
              properties: {
                ...feature.properties,
                visitCount: visitCount,
                hasData: isDepartmentBuilding,
                department_name: isDepartmentBuilding ? buildingName : null
              }
            };
          });

          // Check if source already exists
          if (!map.current.getSource('buildings')) {
            map.current.addSource('buildings', {
              type: 'geojson',
              data: { type: 'FeatureCollection', features: buildingsWithData }
            });
          } else {
            map.current.getSource('buildings').setData({
              type: 'FeatureCollection', 
              features: buildingsWithData 
            });
          }

          // Department colors
          const departmentColors = {
            'CCSICT': '#3498db',      // Blue
            'CCJE': '#2ecc71',        // Green
            'CBM': '#e74c3c',         // Red
            'EDUC': '#9b59b6',        // Purple
            'Agriculture': '#27ae60', // Dark Green
            'Polytechnic': '#34495e', // Dark Gray
            'SAS': '#f39c12',         // Orange
            'Administration_Building': '#95a5a6', // Gray
            'Clinic': '#e67e22',      // Dark Orange
            'Food_Court': '#1abc9c', // Teal
            'Commercial Building': '#7f8c8d' // Light Gray (note: space not underscore)
          };

          // Check if layers already exist
          if (!map.current.getLayer('buildings-fill')) {
            map.current.addLayer({
              id: 'buildings-fill',
              type: 'fill',
              source: 'buildings',
              paint: {
                'fill-color': [
                  'case',
                  ['==', ['get', 'building_name'], 'CCSICT'], departmentColors.CCSICT,
                  ['==', ['get', 'building_name'], 'CCJE'], departmentColors.CCJE,
                  ['==', ['get', 'building_name'], 'CBM'], departmentColors.CBM,
                  ['==', ['get', 'building_name'], 'EDUC'], departmentColors.EDUC,
                  ['==', ['get', 'building_name'], 'Agriculture'], departmentColors.Agriculture,
                  ['==', ['get', 'building_name'], 'Polytechnic'], departmentColors.Polytechnic,
                  ['==', ['get', 'building_name'], 'SAS'], departmentColors.SAS,
                  ['==', ['get', 'building_name'], 'Administration_Building'], departmentColors.Administration_Building,
                  ['==', ['get', 'building_name'], 'Clinic'], departmentColors.Clinic,
                  ['==', ['get', 'building_name'], 'Food_Court'], departmentColors.Food_Court,
                  ['==', ['get', 'building_name'], 'Commercial Building'], departmentColors['Commercial Building'], // Note: space not underscore
                  '#4cceac' // Default color
                ],
                'fill-opacity': 0.7
              }
            });
          }

          if (!map.current.getLayer('buildings-border')) {
            map.current.addLayer({
              id: 'buildings-border',
              type: 'line',
              source: 'buildings',
              paint: {
                'line-color': '#ffffff',
                'line-width': 2
              }
            });
          }

          // Add click events
          map.current.on('click', 'buildings-fill', (e) => {
            const building = e.features[0];
            const buildingName = building.properties.building_name;
            const visitCount = building.properties.visitCount;
            const isDepartment = building.properties.hasData;
            
            let popupContent = `
              <div style="color: #333; font-family: Arial;">
                <h4 style="margin: 0 0 8px 0;">${buildingName.replace('_', ' ')}</h4>
            `;
            
            if (isDepartment && visitCount) {
              popupContent += `
                <p style="margin: 4px 0;"><strong>Clinic Visits:</strong> ${visitCount}</p>
                <p style="margin: 4px 0; font-size: 12px; color: #666;">Department Building</p>
              `;
            } else {
              popupContent += `
                <p style="margin: 4px 0; font-size: 12px; color: #666;">Non-Department Building</p>
                <p style="margin: 4px 0; font-size: 11px; color: #999;">No clinic visit data</p>
              `;
            }
            
            popupContent += `</div>`;
            
            new mapboxgl.Popup()
              .setLngLat(e.lngLat)
              .setHTML(popupContent)
              .addTo(map.current);
          });

          map.current.on('mouseenter', 'buildings-fill', () => {
            map.current.getCanvas().style.cursor = 'pointer';
          });

          map.current.on('mouseleave', 'buildings-fill', () => {
            map.current.getCanvas().style.cursor = '';
          });

          // Add building labels above the fills
          if (!map.current.getLayer('buildings-labels')) {
            map.current.addLayer({
              id: 'buildings-labels',
              type: 'symbol',
              source: 'buildings',
              layout: {
                'text-field': ['get', 'building_name'],
                'text-font': ['Open Sans Semibold', 'Arial Unicode MS Bold'],
                'text-size': 12,
                'text-transform': 'uppercase',
                'text-letter-spacing': 0.1,
                'text-offset': [0, 0],
                'text-anchor': 'center'
              },
              paint: {
                'text-color': '#788490',
                'text-halo-color': '#18191E',
                'text-halo-width': 1
              }
            });
          }

        }

    } catch (error) {
      console.error('Error adding campus features:', error); 
    }
  };

  return (
    <div 
      ref={mapContainer} 
      style={{ 
        width: '100%', 
        height: '100%',
        borderRadius: '8px'
      }} 
    />
  );
};

export default CampusHeatmap;
