import { Box, Button, TextField, useTheme, IconButton, MenuItem, Typography, Alert, Autocomplete } from "@mui/material";
import { Formik } from "formik";
import * as yup from "yup";
import useMediaQuery from "@mui/material/useMediaQuery";
import { useParams, useNavigate } from "react-router-dom";
import { useState, useEffect } from "react";
import CloseIcon from "@mui/icons-material/Close";
import Header from "../../components/Header";
import { tokens } from "../../theme";
import { symptomsAPI, patientsAPI } from "../../services/api";

const SymptomsForm = () => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  const isNonMobile = useMediaQuery("(min-width:600px)");
  const { id } = useParams();
  const navigate = useNavigate();
  const [initialFormValues, setInitialFormValues] = useState(initialValues);
  const [isEdit, setIsEdit] = useState(false);
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  // Load patients and existing data if editing
  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        
        // Load all patients for the optional dropdown
        const patientsResponse = await patientsAPI.getAll();
        const patientsData = patientsResponse.data.results || patientsResponse.data || [];
        console.log('Loaded patients:', patientsData.length, 'patients');
        setPatients(patientsData);
        
        // If editing, load the symptom record
        if (id) {
          setIsEdit(true);
          const symptomResponse = await symptomsAPI.getById(id);
          const symptomData = symptomResponse.data;
          
          setInitialFormValues({
            // Patient info
            existing_patient: symptomData.patient?.id || "",
            patient_name: symptomData.patient?.name || "",
            course: symptomData.patient?.course || "",
            gender: symptomData.patient?.sex || "",
            age: symptomData.patient?.age || "",
            // Symptom info
            date: symptomData.date_logged || new Date().toISOString().split('T')[0],
            symptom: symptomData.symptom || "",
            notes: symptomData.notes || "",
          });
        }
      } catch (error) {
        console.error("Error loading data:", error);
        setError("Failed to load data");
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [id]);

  const handleFormSubmit = async (values, { setSubmitting }) => {
    try {
      setSubmitting(true);
      setError(null);
      setSuccess(null);
      
      console.log('Form submission started with values:', values);
      
      let patientId = values.existing_patient;
      
      // If no existing patient selected, create a new patient
      if (!patientId && values.patient_name) {
        const newPatientData = {
          name: values.patient_name,
          course: values.course,
          sex: values.gender,
          age: parseInt(values.age),
          date_logged: values.date, // Add the date for patient creation
        };
        
        console.log('Creating new patient with data:', newPatientData);
        const patientResponse = await patientsAPI.create(newPatientData);
        patientId = patientResponse.data.id;
        console.log('New patient created with ID:', patientId);
      }

      // Validate required fields
      if (!patientId) {
        throw new Error("Patient ID is required");
      }
      if (!values.symptom) {
        throw new Error("Symptom description is required");
      }

      const submitData = {
        patient: patientId,
        symptom: values.symptom,
        notes: values.notes || "",
        date_logged: new Date(values.date).toISOString(),
      };

      console.group('📤 Preparing Symptom Submission');
      console.log('Patient ID:', patientId);
      console.log('Form values:', values);
      console.log('Final submit data:', submitData);
      console.log('Submit data types:', {
        patient: typeof submitData.patient,
        symptom: typeof submitData.symptom,
        notes: typeof submitData.notes,
        date_logged: typeof submitData.date_logged
      });
      
      // Validate the data before sending
      console.log('Data validation:');
      console.log('- Patient ID is valid:', !!patientId && !isNaN(patientId));
      console.log('- Symptom is not empty:', !!submitData.symptom && submitData.symptom.length > 0);
      console.log('- Date is valid:', !isNaN(new Date(submitData.date_logged).getTime()));
      console.groupEnd();

      if (isEdit) {
        const response = await symptomsAPI.update(id, submitData);
        console.log("Symptom record updated successfully:", response.data);
        setSuccess("Symptom record updated successfully!");
      } else {
        const response = await symptomsAPI.create(submitData);
        console.log("New symptom record created successfully:", response.data);
        setSuccess("New symptom record created successfully!");
      }

      // Wait a moment to show success message, then navigate
      setTimeout(() => {
        navigate("/app/symptoms");
      }, 1500);
    } catch (error) {
      console.group("❌ Form Submission Error");
      console.error("Full error object:", error);
      console.error("Error message:", error.message);
      console.error("Error response:", error.response);
      console.error("Error data:", error.response?.data);
      console.error("Error status:", error.response?.status);
      console.error("Error headers:", error.response?.headers);
      
      // Log the request that failed
      if (error.config) {
        console.error("Failed request config:", {
          url: error.config.url,
          method: error.config.method,
          data: error.config.data,
          headers: error.config.headers
        });
      }
      console.groupEnd();
      
      // More detailed error message
      let errorMessage = "Failed to save symptom record";
      
      if (error.response?.status === 500) {
        errorMessage = "Server Error (500): There's an issue with the backend server. Please check the server logs or contact support.";
      } else if (error.response?.data) {
        if (typeof error.response.data === 'string') {
          errorMessage = error.response.data;
        } else if (error.response.data.detail) {
          errorMessage = error.response.data.detail;
        } else if (error.response.data.error) {
          errorMessage = error.response.data.error;
        } else {
          errorMessage = JSON.stringify(error.response.data);
        }
      }
      
      setError(errorMessage);
    } finally {
      setSubmitting(false);
    }
  };

  const handleClose = () => {
    navigate("/app/symptoms");
  };

  // Course options with display labels and values
  const courseOptions = [
    { label: "STAFF", value: "staff" },
    { label: "OTHERS", value: "others" },
    // College of Laws
    { label: "BSLM", value: "bslm" },
    { label: "JURISDOCTOR", value: "jurisdoctor" },
    // College of Business Management
    { label: "BSBA", value: "bsba" },
    { label: "BSTM", value: "bstm" },
    { label: "BSENTREP", value: "bsentrep" },
    { label: "BSHM", value: "bshm" },
    { label: "BSMA", value: "bsma" },
    { label: "BSAIS", value: "bsais" },
    // College of Criminal Justice Education
    { label: "BSCRIM", value: "bscrim" },
    { label: "BSLEA", value: "bslea" },
    // College of Education
    { label: "BSE", value: "bse" },
    { label: "BEED", value: "beed" },
    { label: "BSED", value: "bsed" },
    { label: "BPED", value: "bped" },
    { label: "CED", value: "ced" },
    { label: "BTVED", value: "btved" },
    { label: "BTLE", value: "btle" },
    // College of Computing Studies, ICT
    { label: "BSCS", value: "bscs" },
    { label: "BSIT", value: "bsit" },
    { label: "BSEMC", value: "bsemc" },
    { label: "BSIS", value: "bsis" },
    { label: "CCSICT", value: "ccsict" },
    // School of Arts & Sciences
    { label: "BAELS", value: "baels" },
    { label: "BAPOS", value: "bapos" },
    { label: "BACOMM", value: "bacomm" },
    { label: "BSBIO", value: "bsbio" },
    { label: "BSMATH", value: "bsmath" },
    { label: "BSCHEM", value: "bschem" },
    { label: "BSPSYCH", value: "bspsych" },
    // Polytechnic School
    { label: "BSITELECTECH", value: "bsitelectech" },
    { label: "BSITAUTOTECH", value: "bsitautotech" },
    { label: "BSINDTECH", value: "bsindtech" },
    { label: "MECHANICALTECH", value: "mechanicaltech" },
    { label: "REFRIGAIRCONDTECH", value: "refrigaircondtech" },
    { label: "ASSOCAIRCRAFTMAINT", value: "assocaircraftmaint" },
    // Agriculture
    { label: "BAT", value: "bat" },
    { label: "BSAGRI", value: "bsagri" },
    { label: "BSAGRIBIZ", value: "bsagribiz" },
    { label: "BSENVI", value: "bsenvi" },
    { label: "BSFOR", value: "bsfor" },
    { label: "BSFISHERIES", value: "bsfisheries" },
    // Graduate Programs
    { label: "DIT", value: "dit" },
    { label: "MIT", value: "mit" },
    { label: "DDSA", value: "ddsa" },
    { label: "MBA", value: "mba" },
    { label: "MPA", value: "mpa" },
    { label: "MASTERLAWS", value: "masterlaws" },
    { label: "MAED", value: "maed" },
    { label: "PHD_ED", value: "phd_ed" },
    { label: "PHD_ANIMAL", value: "phd_animal" },
    { label: "PHD_CROP", value: "phd_crop" },
  ];

  if (loading) {
    return (
      <Box m="20px" display="flex" justifyContent="center" alignItems="center" height="400px">
        <div>Loading...</div>
      </Box>
    );
  }

  return (
    <Box m="20px">
      {/* Header with Close Button */}
      <Box
        display="flex"
        justifyContent="space-between"
        alignItems="center"
        mb="20px"
      >
        <Header
          title={isEdit ? "EDIT SYMPTOM RECORD" : "CREATE SYMPTOM RECORD"}
          subtitle={isEdit ? "Update existing symptom information" : "Add new student health record"}
        />
        <IconButton onClick={handleClose}>
          <CloseIcon />
        </IconButton>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {success && (
        <Alert severity="success" sx={{ mb: 2 }}>
          {success}
        </Alert>
      )}

      <Formik
        onSubmit={handleFormSubmit}
        initialValues={initialFormValues}
        validationSchema={checkoutSchema}
        enableReinitialize={true}
      >
        {({
          values,
          errors,
          touched,
          handleBlur,
          handleChange,
          handleSubmit,
          isSubmitting,
          setFieldValue,
        }) => (
          <form onSubmit={handleSubmit}>
            <Box
              display="grid"
              gap="30px"
              gridTemplateColumns="repeat(4, minmax(0, 1fr))"
              sx={{
                "& > div": { gridColumn: isNonMobile ? undefined : "span 4" },
              }}
            >
              {/* Date */}
              <TextField
                fullWidth
                variant="filled"
                type="date"
                label="Date"
                onBlur={handleBlur}
                onChange={handleChange}
                value={values.date}
                name="date"
                error={!!touched.date && !!errors.date}
                helperText={touched.date && errors.date}
                sx={{ gridColumn: "span 2" }}
                autoComplete="off"
                InputLabelProps={{
                  shrink: true,
                }}
              />

              {/* Optional: Select Existing Patient */}
              <Autocomplete
                options={[{ id: "", name: "Enter new patient details below", course: "", age: "" }, ...patients]}
                getOptionLabel={(option) => 
                  option.id === "" ? option.name : `${option.name} - ${option.course} - Age ${option.age}`
                }
                value={patients.find(p => p.id === values.existing_patient) || null}
                onChange={(event, newValue) => {
                  setFieldValue('existing_patient', newValue?.id || "");
                  if (newValue && newValue.id !== "") {
                    setFieldValue('patient_name', newValue.name);
                    setFieldValue('course', newValue.course);
                    setFieldValue('gender', newValue.sex);
                    setFieldValue('age', newValue.age);
                  }
                }}
                renderInput={(params) => (
                  <TextField
                    {...params}
                    fullWidth
                    variant="filled"
                    label="Select Existing Patient (Optional)"
                    autoComplete="off"
                    sx={{ gridColumn: "span 2" }}
                  />
                )}
                sx={{ gridColumn: "span 2" }}
              />

              {/* Patient Name */}
              <TextField
                fullWidth
                variant="filled"
                type="text"
                label="Patient Name"
                onBlur={handleBlur}
                onChange={handleChange}
                value={values.patient_name}
                name="patient_name"
                error={!!touched.patient_name && !!errors.patient_name}
                helperText={touched.patient_name && errors.patient_name}
                sx={{ gridColumn: "span 2" }}
                autoComplete="off"
              />

              {/* Course */}
              <Autocomplete
                options={courseOptions}
                getOptionLabel={(option) => option.label}
                value={courseOptions.find(option => option.value === values.course) || null}
                onChange={(event, newValue) => {
                  setFieldValue('course', newValue?.value || "");
                }}
                renderInput={(params) => (
                  <TextField
                    {...params}
                    fullWidth
                    variant="filled"
                    label="Course"
                    name="course"
                    error={!!touched.course && !!errors.course}
                    helperText={touched.course && errors.course}
                    autoComplete="off"
                  />
                )}
                sx={{ gridColumn: "span 2" }}
              />

              {/* Gender */}
              <TextField
                fullWidth
                variant="filled"
                select
                label="Gender"
                onBlur={handleBlur}
                onChange={handleChange}
                value={values.gender}
                name="gender"
                error={!!touched.gender && !!errors.gender}
                helperText={touched.gender && errors.gender}
                sx={{ gridColumn: "span 2" }}
                autoComplete="off"
              >
                <MenuItem value="Male">Male</MenuItem>
                <MenuItem value="Female">Female</MenuItem>
                <MenuItem value="Other">Other</MenuItem>
                <MenuItem value="Prefer not to say">Prefer not to say</MenuItem>
              </TextField>

              {/* Age */}
              <TextField
                fullWidth
                variant="filled"
                type="number"
                label="Age"
                onBlur={handleBlur}
                onChange={handleChange}
                value={values.age}
                name="age"
                error={!!touched.age && !!errors.age}
                helperText={touched.age && errors.age}
                sx={{ gridColumn: "span 2" }}
                autoComplete="off"
                inputProps={{ min: 16, max: 100 }}
              />

              {/* Symptom Description */}
              <TextField
                fullWidth
                variant="filled"
                type="text"
                label="Symptoms"
                onBlur={handleBlur}
                onChange={handleChange}
                value={values.symptom}
                name="symptom"
                error={!!touched.symptom && !!errors.symptom}
                helperText={touched.symptom && errors.symptom}
                sx={{ gridColumn: "span 4" }}
                autoComplete="off"
                multiline
                rows={4}
                placeholder="Describe the symptoms experienced by the student..."
              />

              {/* Notes */}
              <TextField
                fullWidth
                variant="filled"
                type="text"
                label="Additional Notes"
                onBlur={handleBlur}
                onChange={handleChange}
                value={values.notes}
                name="notes"
                error={!!touched.notes && !!errors.notes}
                helperText={touched.notes && errors.notes}
                sx={{ gridColumn: "span 4" }}
                autoComplete="off"
                multiline
                rows={3}
                placeholder="Additional notes, recommendations, or observations..."
              />
            </Box>

            <Box display="flex" justifyContent="end" mt="20px" gap="10px">
              <Button
                type="button"
                color="secondary"
                variant="contained"
                onClick={handleClose}
              >
                Cancel
              </Button>
              <Button
                type="submit"
                color="secondary"
                variant="contained"
                disabled={isSubmitting}
              >
                {isSubmitting ? "Saving..." : (isEdit ? "Update Record" : "Create Record")}
              </Button>
            </Box>
          </form>
        )}
      </Formik>
    </Box>
  );
};

// Validation Schema
const checkoutSchema = yup.object().shape({
  date: yup.date().required("Date is required"),
  patient_name: yup.string().required("Patient name is required"),
  course: yup.string().required("Course is required"),
  gender: yup.string().required("Gender is required"),
  age: yup
    .number()
    .min(16, "Age must be at least 16")
    .max(100, "Age must be less than 100")
    .required("Age is required"),
  symptom: yup
    .string()
    .min(5, "Please provide more detailed symptoms")
    .required("Symptom description is required"),
  notes: yup.string(), // Optional field
});

// Initial Values
const initialValues = {
  date: new Date().toISOString().split('T')[0], // Today's date
  existing_patient: "",
  patient_name: "",
  course: "",
  gender: "",
  age: "",
  symptom: "",
  notes: "",
};

export default SymptomsForm;