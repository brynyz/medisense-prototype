import { Box, IconButton, useTheme, Paper, List, ListItem, ListItemIcon, ListItemText, ClickAwayListener } from "@mui/material";
import { Link, useNavigate } from "react-router-dom";
import { useContext, useState, useRef } from "react";
import { ColorModeContext, tokens } from "../../theme";
import InputBase from "@mui/material/InputBase";
import LightModeOutlinedIcon from "@mui/icons-material/LightModeOutlined";
import DarkModeOutlinedIcon from "@mui/icons-material/DarkModeOutlined";
import NotificationsOutlinedIcon from "@mui/icons-material/NotificationsOutlined";
import SettingsOutlinedIcon from "@mui/icons-material/SettingsOutlined";
import PersonOutlinedIcon from "@mui/icons-material/PersonOutlined";
import SearchIcon from "@mui/icons-material/Search";
import HomeOutlinedIcon from "@mui/icons-material/HomeOutlined";
import LocalHospitalIcon from "@mui/icons-material/LocalHospital";
import DatasetIcon from "@mui/icons-material/Dataset";
import DataExplorationIcon from "@mui/icons-material/DataExploration";
import CalendarTodayOutlinedIcon from "@mui/icons-material/CalendarTodayOutlined";
import AssistantIcon from "@mui/icons-material/Assistant";
import HelpOutlineOutlinedIcon from "@mui/icons-material/HelpOutlineOutlined";
import MapIcon from "@mui/icons-material/Map";

const Topbar = () => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  const colorMode = useContext(ColorModeContext);
  const navigate = useNavigate();
  
  const [searchQuery, setSearchQuery] = useState("");
  const [showResults, setShowResults] = useState(false);
  const searchRef = useRef(null);

  // Define searchable pages with correct paths
  const pages = [
    { title: "Dashboard", path: "/dashboard", icon: <HomeOutlinedIcon />, keywords: ["dashboard", "home", "analytics", "overview"] },
    { title: "Symptoms", path: "/app/symptoms", icon: <LocalHospitalIcon />, keywords: ["symptoms", "medical", "health", "patient", "clinic"] },
    { title: "Data Preprocessing", path: "/app/datapreprocessing", icon: <DatasetIcon />, keywords: ["data", "preprocessing", "clean", "process"] },
    { title: "Model Training", path: "/app/modeltraining", icon: <DataExplorationIcon />, keywords: ["model", "training", "machine learning", "ml", "ai"] },
    { title: "Calendar", path: "/app/calendar", icon: <CalendarTodayOutlinedIcon />, keywords: ["calendar", "schedule", "events", "appointments"] },
    { title: "Smart Help", path: "/app/help", icon: <AssistantIcon />, keywords: ["help", "assistant", "support", "guide"] },
    { title: "FAQ", path: "/app/faq", icon: <HelpOutlineOutlinedIcon />, keywords: ["faq", "questions", "answers", "help"] },
    { title: "Settings", path: "/app/profile", icon: <SettingsOutlinedIcon />, keywords: ["settings", "profile", "account", "preferences"] },
    { title: "heatmap", path: "/app/heatmap", icon: <MapIcon />, keywords: ["heatmap", "map", "visualization", "geography"] },
  ];

  // Filter pages based on search query
  const filteredPages = pages.filter(page => 
    page.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    page.keywords.some(keyword => keyword.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  const handleSearchChange = (event) => {
    const value = event.target.value;
    setSearchQuery(value);
    setShowResults(value.length > 0);
  };

  const handlePageSelect = (path) => {
    console.log('Navigating to:', path); // Debug log
    navigate(path);
    setSearchQuery("");
    setShowResults(false);
  };

  const handleSearchSubmit = (event) => {
    event.preventDefault();
    if (filteredPages.length > 0) {
      console.log('Form submitted, navigating to:', filteredPages[0].path); // Debug log
      handlePageSelect(filteredPages[0].path);
    }
  };

  const handleClickAway = () => {
    setShowResults(false);
  };

  return (
    <Box display="flex" justifyContent="space-between" p={2}>
      {/* SEARCH BAR */}
      <ClickAwayListener onClickAway={handleClickAway}>
        <Box position="relative" ref={searchRef}>
          <Box
            component="form"
            onSubmit={handleSearchSubmit}
            display="flex"
            backgroundColor={colors.primary[400]}
            borderRadius="3px"
            width="300px"
          >
            <InputBase 
              sx={{ ml: 2, flex: 1 }} 
              placeholder="Search pages..." 
              value={searchQuery}
              onChange={handleSearchChange}
              onFocus={() => searchQuery && setShowResults(true)}
            />
            <IconButton type="submit" sx={{ p: 1 }}>
              <SearchIcon />
            </IconButton>
          </Box>

          {/* Search Results Dropdown */}
          {showResults && filteredPages.length > 0 && (
            <Paper
              sx={{
                position: "absolute",
                top: "100%",
                left: 0,
                right: 0,
                zIndex: 1000,
                maxHeight: "300px",
                overflow: "auto",
                backgroundColor: colors.primary[400],
                borderRadius: "4px",
                mt: 1
              }}
            >
              <List sx={{ py: 0 }}>
                {filteredPages.slice(0, 6).map((page, index) => (
                  <ListItem
                    key={page.path}
                    button
                    onClick={() => handlePageSelect(page.path)}
                    sx={{
                      "&:hover": {
                        backgroundColor: colors.primary[300],
                      },
                      borderBottom: index < filteredPages.length - 1 ? `1px solid ${colors.grey[700]}` : "none"
                    }}
                  >
                    <ListItemIcon sx={{ color: colors.grey[100], minWidth: "35px" }}>
                      {page.icon}
                    </ListItemIcon>
                    <ListItemText 
                      primary={page.title}
                      sx={{ 
                        "& .MuiListItemText-primary": { 
                          color: colors.grey[100],
                          fontSize: "14px"
                        }
                      }}
                    />
                  </ListItem>
                ))}
              </List>
            </Paper>
          )}

          {/* No Results */}
          {showResults && filteredPages.length === 0 && searchQuery && (
            <Paper
              sx={{
                position: "absolute",
                top: "100%",
                left: 0,
                right: 0,
                zIndex: 1000,
                backgroundColor: colors.primary[400],
                border: `1px solid ${colors.grey[700]}`,
                borderRadius: "4px",
                mt: 1,
                p: 2
              }}
            >
              <Box color={colors.grey[300]} fontSize="14px">
                No pages found for "{searchQuery}"
              </Box>
            </Paper>
          )}
        </Box>
      </ClickAwayListener>

      {/* ICONS */}
      <Box display="flex">
        <IconButton onClick={colorMode.toggleColorMode}>
          {theme.palette.mode === "dark" ? (
            <DarkModeOutlinedIcon />
          ) : (
            <LightModeOutlinedIcon />
          )}
        </IconButton>
        <IconButton
          component={Link}
          to="/app/profile"
        >
          <SettingsOutlinedIcon />
        </IconButton>
        <IconButton
          component={Link}
          to="/app/profile"
        >
          <PersonOutlinedIcon />
        </IconButton>
      </Box>
    </Box>
  );
};

export default Topbar;