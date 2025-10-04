import { Box, Button, Typography, useTheme } from "@mui/material";
import { DataGrid } from "@mui/x-data-grid";
import { tokens } from "../../theme";
import Header from "../../components/Header";
import { useNavigate } from "react-router-dom";
import { useState, useEffect, useCallback } from 'react';
import { symptomsAPI } from '../../services/api';
import EditOutlinedIcon from "@mui/icons-material/EditOutlined";
import DeleteOutlinedIcon from "@mui/icons-material/DeleteOutlined";
import VisibilityOutlinedIcon from "@mui/icons-material/VisibilityOutlined";
import AddIcon from "@mui/icons-material/Add";
import { CircularProgress, Alert, IconButton } from "@mui/material";

// Transform Django API data to match DataGrid format
const transformApiData = (apiData) => {
  // Handle different response formats
  let dataArray = apiData;
  
  // If it's a paginated response, get the results array
  if (apiData && typeof apiData === 'object' && apiData.results) {
    dataArray = apiData.results;
  }
  
  // If it's not an array, wrap it in an array or return empty array
  if (!Array.isArray(dataArray)) {
    console.warn('API data is not an array:', apiData);
    return [];
  }
  
  return dataArray.map((item) => ({
    id: item.id,
    date: item.date_logged || item.patient?.date_logged || null,
    course: item.patient?.course || 'N/A',
    gender: item.patient?.sex || 'N/A', 
    age: item.patient?.age || 'N/A',
    symptoms: item.symptom || 'N/A',
    notes: item.notes || '',
    patient_name: item.patient?.name || item.patient_name || `Patient ${item.patient}` || 'Unknown',
    patient_id: item.patient?.id || item.patient || null,
  }));
};

const Symptoms = () => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  const navigate = useNavigate();
  const [symptomsData, setSymptomsData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Load symptoms data from Django API
  const loadSymptomsData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await symptomsAPI.getAll();
      console.log('Symptoms API Response:', response.status, response.data);
      
      const transformedData = transformApiData(response.data);
      console.log('Transformed symptoms data:', transformedData.length, 'records');
      
      setSymptomsData(transformedData);
    } catch (error) {
      console.error('Error loading symptoms data:', error);
      console.error('Error status:', error.response?.status);
      console.error('Error data:', error.response?.data);
      console.error('Request headers:', error.config?.headers);
      
      setError(error.response?.data?.detail || error.message || 'Failed to load symptoms data');
      
      // Only redirect to login if it's actually a 401 error
      if (error.response?.status === 401) {
        console.log('401 error - token might be invalid or missing');
        console.log('Clearing tokens and redirecting to login');
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        navigate('/login');
      } else {
        console.log('Non-401 error, not redirecting to login');
      }
    } finally {
      setLoading(false);
    }
  }, [navigate]);

  // Load data on component mount
  useEffect(() => {
    loadSymptomsData();
  }, []);

  // Handle create action
  const handleCreate = () => {
    navigate("/app/symptoms/create");
  };

  // Handle edit action
  const handleEdit = (id) => {
    navigate(`/app/symptoms/edit/${id}`);
  };

  // Handle delete action
  const handleDelete = async (id) => {
    if (window.confirm("Are you sure you want to delete this symptom record?")) {
      try {
        await symptomsAPI.delete(id);
        // Reload data after successful deletion
        await loadSymptomsData();
        console.log("Symptom record deleted successfully");
      } catch (error) {
        console.error("Error deleting symptom record:", error);
        setError(error.response?.data?.detail || 'Failed to delete symptom record');
      }
    }
  };

  // Handle view action
  const handleView = (id) => {
    console.log("View symptom record with ID:", id);
    navigate(`/app/symptoms/view/${id}`);
  };

  const columns = [
    { field: "id", headerName: "ID", width: 50 },
    {
      field: "date",
      headerName: "Date",
      width: 120,
      renderCell: (params) => {
        if (!params.value) {
          return <Typography>-</Typography>;
        }
        
        try {
          const date = new Date(params.value);
          if (isNaN(date.getTime())) {
            return <Typography>Invalid Date</Typography>;
          }
          return (
            <Typography>
              {date.toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'short',
                day: 'numeric'
              })}
            </Typography>
          );
        } catch (error) {
          return <Typography>-</Typography>;
        }
      },
    },
    {
      field: "patient_name",
      headerName: "Patient",
      flex: 0.8,
      cellClassName: "name-column--cell",
    },
    {
      field: "course",
      headerName: "Course",
      flex: 0.5,
      cellClassName: "course-column--cell",
    },
    {
      field: "gender",
      headerName: "Gender",
      width: 100,
      renderCell: (params) => (
        <Typography
          sx={{
            color: params.row.gender === "Male" 
              ? colors.blueAccent[500] 
              : colors.redAccent[500],
            fontWeight: "500",
          }}
        >
          {params.row.gender}
        </Typography>
      ),
    },
    {
      field: "age",
      headerName: "Age",
      type: "number",
      headerAlign: "left",
      align: "left",
      width: 80,
      renderCell: (params) => (
        <Typography>
          {params.row.age || 'N/A'}
        </Typography>
      ),
    },
    {
      field: "symptoms",
      headerName: "Symptoms",
      flex: 2,
      renderCell: (params) => (
        <Box
          sx={{
            whiteSpace: 'normal',
            wordWrap: 'break-word',
            lineHeight: 1.2,
            py: 1,
          }}
        >
          <Typography variant="body2">
            {params.row.symptoms}
          </Typography>
        </Box>
      ),
    },
    {
      field: "notes",
      headerName: "Notes",
      flex: 1.5,
      renderCell: (params) => (
        <Box
          sx={{
            whiteSpace: 'normal',
            wordWrap: 'break-word',
            lineHeight: 1.2,
            py: 1,
          }}
        >
          <Typography variant="body2" color="textSecondary">
            {params.row.notes || 'No notes'}
          </Typography>
        </Box>
      ),
    },
    {
      field: "actions",
      headerName: "Actions",
      width: 120,
      sortable: false,
      renderCell: (params) => {
        return (
          <Box display="flex" gap={0.5}>
            <IconButton
              onClick={() => handleView(params.row.id)}
              sx={{ color: colors.greenAccent[400] }}
              size="small"
            >
              <VisibilityOutlinedIcon />
            </IconButton>
            
            <IconButton
              onClick={() => handleEdit(params.row.id)}
              sx={{ color: colors.blueAccent[400] }}
              size="small"
            >
              <EditOutlinedIcon />
            </IconButton>
            <IconButton
              onClick={() => handleDelete(params.row.id)}
              sx={{ color: colors.redAccent[400] }}
              size="small"
            >
              <DeleteOutlinedIcon />
            </IconButton>
          </Box>
        );
      },
    },
  ];

  if (loading) {
    return (
      <Box m="20px">
        <Header title="SYMPTOMS" subtitle="Loading symptoms data..." />
        <Box display="flex" justifyContent="center" mt="20px">
          <CircularProgress />
        </Box>
      </Box>
    );
  }

  if (error) {
    return (
      <Box m="20px">
        <Header title="SYMPTOMS" subtitle="Error loading data" />
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
          <Button 
            onClick={loadSymptomsData} 
            sx={{ ml: 2 }}
            variant="outlined"
          >
            Retry
          </Button>
        </Alert>
      </Box>
    );
  }

  return (
    <Box m="20px">
      {/* Header with Create Button */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb="20px">
        <Header title="SYMPTOMS" subtitle="Student Health Symptoms Records" />
        <Button
          onClick={handleCreate}
          variant="contained"
          startIcon={<AddIcon />}
          sx={{
            backgroundColor: colors.greenAccent[600],
            color: colors.grey[100],
            fontSize: "14px",
            fontWeight: "bold",
            padding: "10px 20px",
            "&:hover": {
              backgroundColor: colors.greenAccent[700],
            },
          }}
        >
          Add Symptom Record
        </Button>
      </Box>

      {/* DataGrid */}
      <Box
        m="20px 0 0 0"
        height="75vh"
        sx={{
          "& .MuiDataGrid-root": {
            border: "none",
          },
          "& .MuiDataGrid-cell": {
            borderBottom: "none",
            display: "flex",
            alignItems: "center",
          },
          "& .course-column--cell": {
            color: colors.greenAccent[300],
          },
          "& .name-column--cell": {
            color: colors.blueAccent[300],
            fontWeight: "600",
          },
          "& .MuiDataGrid-columnHeaders": {
            backgroundColor: colors.blueAccent[700],
            borderBottom: "none",
          },
          "& .MuiDataGrid-virtualScroller": {
            backgroundColor: colors.primary[400],
          },
          "& .MuiDataGrid-footerContainer": {
            borderTop: "none",
            backgroundColor: colors.blueAccent[700],
          },
          "& .MuiCheckbox-root": {
            color: `${colors.greenAccent[200]} !important`,
          },
          "& .MuiDataGrid-pagination": {
            color: colors.grey[100],
          },
          "& .MuiTablePagination-root": {
            color: colors.grey[100],
          },
          "& .MuiTablePagination-selectIcon": {
            color: colors.grey[100],
          },
          "& .MuiIconButton-root": {
            color: colors.grey[100],
          },
        }}
      >
        <DataGrid
          checkboxSelection
          rows={symptomsData}
          columns={columns}
          pagination
          paginationMode="client"
          pageSizeOptions={[5, 10, 25, 50, 100]}
          initialState={{
            pagination: {
              paginationModel: {
                pageSize: 25,
              },
            },
          }}
          disableRowSelectionOnClick
          getRowHeight={() => 80}
          sx={{
            width: '100%',
            '& .MuiDataGrid-root': {
              width: '100%',
            },
          }}
        />
      </Box>
    </Box>
  );
};

export default Symptoms;