import { Box, useTheme } from "@mui/material";
import CampusMapHeatmap from "../../components/CampusHeatmap";
import Header from "../../components/Header";
import { tokens } from "../../theme";
import { Link } from "react-router-dom";
import FullScreenExitIcon from "@mui/icons-material/FullscreenExit";
import IconButton from "@mui/material/IconButton";
import Dashboard from "../dashboard";

const Heatmap = () => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);

  return (
    <Box m="20px">
      <Box
        mt="8px"
        p="0 30px"
        display="flex "
        justifyContent="space-between"
        alignItems="center"
      >
        <Box>
        <Header
        title="Campus Visit Heatmap"
        subtitle="Department-based clinic visit visualization with interactive features"
      />
        </Box>
        <Box>
          <IconButton component={Link} to="/dashboard">
            <FullScreenExitIcon
              sx={{ fontSize: "26px", color: colors.greenAccent[500] }}
            />
          </IconButton>
        </Box>
      </Box>

      <Box
        height="75vh"
        border={`1px solid ${colors.grey[100]}`}
        borderRadius="4px"
        overflow="hidden"
      >
        <CampusMapHeatmap isDashboard={false} />
      </Box>
    </Box>
  );
};

export default Heatmap;
