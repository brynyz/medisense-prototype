import { Box, useTheme, Paper } from "@mui/material";
import { useEffect } from "react";
import { tokens } from "../../theme";
import Header from "../../components/Header";

const Help = () => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);

  useEffect(() => {
    // Configure ConvoCore
    window.VG_CONFIG = {
      ID: "3lq7EEafoOHg1Gy33RAm",
      region: 'na', // 'eu' or 'na' corresponding to Europe and North America
      render: 'full-width', // popup or full-width
      // modalMode: true, // Set this to 'true' to open the widget in modal mode
      stylesheets: [
        // Base CONVOCORE CSS
        "https://vg-bunny-cdn.b-cdn.net/vg_live_build/styles.css",
        // Add your custom css stylesheets if needed
      ],
      // Optional: Add user data if available
      // user: {
      //   name: 'John Doe', // User's name
      //   email: 'johndoe@gmail.com', // User's email
      //   phone: '+1234567890', // User's phone number
      // },
      // userID: 'USER_ID', // If you want to use your own user_id
      // autostart: true, // Whether to autostart the chatbot with the proactive message
    };

    // Load ConvoCore script
    const script = document.createElement("script");
    script.src = "https://vg-bunny-cdn.b-cdn.net/vg_live_build/vg_bundle.js";
    script.defer = true; // Remove 'defer' if you want widget to load faster
    document.body.appendChild(script);

    // Cleanup function
    return () => {
      // Remove script when component unmounts
      if (document.body.contains(script)) {
        document.body.removeChild(script);
      }
      // Clean up global config
      delete window.VG_CONFIG;
    };
  }, []);

  return (
    <Box m="20px">
      <Header title="HELP & SUPPORT" subtitle="Get assistance with MediSense Support" />
      <Paper
        elevation={3}
        sx={{
          backgroundColor: "transparent",
          borderRadius: "2px",
          marginTop: "10px",
        //   marginLeft: "100px",
        //   marginRight: "100px",
        }}
      >
        {/* ConvoCore Container */}
        <Box
          id="VG_OVERLAY_CONTAINER"
          sx={{
            width: "100%",
            height: "70vh", // Responsive height
            minHeight: "750px",
            backgroundColor: colors.primary[300],
            // borderRadius: "8px",
            border: `1px solid ${colors.grey[700]}`,
          }}
        >
          {/* ConvoCore will render here */}
        </Box>
      </Paper>
    </Box>
  );
};

export default Help;