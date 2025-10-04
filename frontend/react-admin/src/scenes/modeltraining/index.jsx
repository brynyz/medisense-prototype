import { Box, useTheme, Paper } from "@mui/material";
import { tokens } from "../../theme";
import Header from "../../components/Header";

const ModelTraining = () => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);

  return (
    <Box ml="20px" mr ="20px" mb = "8px">
      <Header
        title="MODEL DASHBOARD"
        subtitle="Machine Learning Model Training Interface"
      />

      <Box mt="10px" mb="0px">
        <Paper
          elevation={3}
          sx={{
            backgroundColor: colors.primary[400],
            borderRadius: "8px",
            overflow: "hidden",
            height: "80vh", // Adjust height as needed
          }}
        >
          <iframe
            src="http://localhost:8501" // Your Streamlit app URL
            width="100%"
            height="100%"
            style={{
              border: "none",
              borderRadius: "8px",
            }}
            title="Model Training Dashboard"
            allowFullScreen
            // Optional: Add additional permissions if needed
            allow="camera; microphone; fullscreen"
          />
        </Paper>
      </Box>
    </Box>
  );
};

export default ModelTraining;