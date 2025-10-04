import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { geoFeatures } from "../data/isuGeoFeatures";

// Fix for default markers in Leaflet with webpack
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

const OpenStreetMapComponent = ({ isDashboard = false }) => {
  const mapRef = useRef(null);
  const mapInstance = useRef(null);

  useEffect(() => {
    if (mapInstance.current) return;

    // Initialize map
    mapInstance.current = L.map(mapRef.current).setView([16.938, 121.7645], 17);

    // Add OpenStreetMap tiles
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors'
    }).addTo(mapInstance.current);

    // Add campus boundary
    const campusBoundary = geoFeatures.features.filter(
      feature => feature.properties.reference === "initial_zoom"
    );

    if (campusBoundary.length > 0) {
      L.geoJSON(campusBoundary, {
        style: {
          color: '#ff0000',
          weight: 2,
          fillOpacity: 0,
          dashArray: '5, 5'
        }
      }).addTo(mapInstance.current);
    }

    // Add buildings
    const buildings = geoFeatures.features.filter(
      feature => feature.properties.building_name && !feature.properties.general_area
    );

    const departmentColors = {
      'CCSICT': '#3498db',
      'CCJE': '#2ecc71',
      'CBM': '#e74c3c',
      'EDUC': '#9b59b6',
      'Agriculture': '#27ae60',
      'Polytechnic': '#34495e',
      'SAS': '#f39c12',
      'Administration_Building': '#95a5a6',
      'Clinic': '#e67e22',
      'Food_Court': '#1abc9c',
      'Commercial Building': '#7f8c8d'
    };

    buildings.forEach(building => {
      const buildingName = building.properties.building_name;
      const color = departmentColors[buildingName] || '#4cceac';
      
      L.geoJSON(building, {
        style: {
          color: '#ffffff',
          weight: 2,
          fillColor: color,
          fillOpacity: 0.7
        }
      }).bindPopup(`
        <div>
          <strong>${buildingName.replace('_', ' ')}</strong><br/>
          <em>Click for visit details</em><br/>
          <small>Visits: ${Math.floor(Math.random() * 100)}</small>
        </div>
      `).addTo(mapInstance.current);
    });

    return () => {
      if (mapInstance.current) {
        mapInstance.current.remove();
        mapInstance.current = null;
      }
    };
  }, []);

  return (
    <div 
      ref={mapRef} 
      style={{ 
        width: '100%', 
        height: isDashboard ? '300px' : '500px',
        borderRadius: '8px'
      }} 
    />
  );
};

export default OpenStreetMapComponent;
