import { useTheme } from "@mui/material";
import { ResponsiveGeoMap } from "@nivo/geo";
import { geoFeatures } from "../data/isuGeoFeatures";
import { tokens } from "../theme";

const GeographyChart = ({ isDashboard = false, data = [] }) => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);

  // Get the campus boundary (initial_zoom reference)
  const campusBoundary = geoFeatures.features.filter(
    feature => feature.properties.reference === "initial_zoom"
  );

  // Filter features to show only buildings (not general areas)
  const buildingFeatures = geoFeatures.features.filter(
    feature => feature.properties.building_name && !feature.properties.general_area
  );

  // Combine campus boundary and buildings
  const allFeatures = [...campusBoundary, ...buildingFeatures];

  return (
    <ResponsiveGeoMap
      features={allFeatures}
      margin={{ top: 0, right: 0, bottom: 0, left: 0 }}
      projectionScale={isDashboard ? 8000000 : 12000000}
      projectionTranslation={[0.5, 0.5]}
      projectionRotation={[-121.7645, -16.938, -78]}
      fillColor={(feature) => {
        //Campus boundary - transparent with border only
        if (feature.properties.reference === "initial_zoom") {
          return "transparent";
        }
        // buildings - filled color
        return "#4cceac";
      }}
      borderWidth={(feature) => {
        //Campus boundary - thicker border
        if (feature.properties.reference === "initial_zoom") {
          return 2;
        }
        // buildings - normal border
        return 1;
      }}
      borderColor={(feature) => {
        //Campus boundary - distinct color
        if (feature.properties.reference === "initial_zoom") {
          return colors.redAccent[500];
        }
        // buildings - white border
        return "#ffffff";
      }}
      enableGraticule={false}
      theme={{
        background: "transparent",
        text: {
          fill: colors.grey[100],
        },
        tooltip: {
          container: {
            background: colors.primary[400],
            color: colors.grey[100],
          },
        },
      }}
      tooltip={({ feature }) => (
        <div
          style={{
            background: colors.primary[400],
            padding: "9px 12px",
            border: `1px solid ${colors.grey[100]}`,
            borderRadius: "4px",
            color: colors.grey[100],
          }}
        >
          <strong>
            {feature.properties.reference === "initial_zoom" 
              ? "Campus Boundary" 
              : (feature.properties.building_name || "Campus Area")
            }
          </strong>
          {data && data.length > 0 && feature.properties.building_name && (
            <div>Visits: {Math.floor(Math.random() * 50)}</div>
          )}
        </div>
      )}
    />
  );
};

export default GeographyChart;