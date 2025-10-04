import { Box, useTheme } from "@mui/material";
import Header from "../../components/Header";
import Accordion from "@mui/material/Accordion";
import AccordionSummary from "@mui/material/AccordionSummary";
import AccordionDetails from "@mui/material/AccordionDetails";
import Typography from "@mui/material/Typography";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import { tokens } from "../../theme";
import { useState } from "react"; // Add this import

const FAQ = () => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  
  // State to track which accordion is open
  const [expandedPanel, setExpandedPanel] = useState(false);

  // Handle accordion change
  const handleAccordionChange = (panel) => (event, isExpanded) => {
    setExpandedPanel(isExpanded ? panel : false);
  };

  return (
    <Box m="20px">
      <Header title="FAQ" subtitle="Frequently Asked Questions Page" />

      <Typography
        mb="10px"
        fontWeight={600}
        variant="h4"
        color={colors.grey[100]}
      >
        General Questions
      </Typography>

      <Accordion 
        expanded={expandedPanel === 'panel1'} 
        onChange={handleAccordionChange('panel1')}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography color={colors.greenAccent[500]} variant="h5">
            What is Medisense?
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Typography>
            Medisense is a decision-support system for campus clinics; currently
            localized in Isabela State University - Cayuayan City Campus. It
            leverages machine learning to predict medical trends using
            historical patient data and provides periodic medicine
            recommendations based on these predictions to ensure availability
            and minimizing wastage.
          </Typography>
        </AccordionDetails>
      </Accordion>

      <Accordion 
        expanded={expandedPanel === 'panel2'} 
        onChange={handleAccordionChange('panel2')}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography color={colors.greenAccent[500]} variant="h5">
            Who can use Medisense?
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Typography>
            It is designed for clinic staff, administrators, and researchers who
            need insights on patient trends and medicine stock.
          </Typography>
        </AccordionDetails>
      </Accordion>

      <Typography
        mb="10px"
        mt="20px"
        fontWeight={600}
        variant="h4"
        color={colors.grey[100]}
      >
        Data Privacy & Security
      </Typography>

      <Accordion 
        expanded={expandedPanel === 'panel3'} 
        onChange={handleAccordionChange('panel3')}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography color={colors.greenAccent[500]} variant="h5">
            Are my data secure with Medisense?
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Typography>
            All data is encrypted and stored securely. We comply with data
            privacy regulations to ensure that the collected data is protected
            and used responsibly.
          </Typography>
        </AccordionDetails>
      </Accordion>

      <Accordion 
        expanded={expandedPanel === 'panel4'} 
        onChange={handleAccordionChange('panel4')}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography color={colors.greenAccent[500]} variant="h5">
            What measures are in place to protect patient confidentiality?
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Typography>
            Every clinic record is anonymized and access is restricted to
            authorized personnel only.
          </Typography>
        </AccordionDetails>
      </Accordion>

      <Typography
        mb="10px"
        mt="20px"
        fontWeight={600}
        variant="h4"
        color={colors.grey[100]}
      >
        Data Preprocessing & Input
      </Typography>

      <Accordion 
        expanded={expandedPanel === 'panel5'} 
        onChange={handleAccordionChange('panel5')}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography color={colors.greenAccent[500]} variant="h5">
            What happens if a symptom is not listed?
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Typography>
            If a symptom is not listed in our predefined options, you can add it as a custom entry in the "Other" field or contact the system administrator to have it added to the main symptom database.
          </Typography>
        </AccordionDetails>
      </Accordion>

      <Accordion 
        expanded={expandedPanel === 'panel6'} 
        onChange={handleAccordionChange('panel6')}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography color={colors.greenAccent[500]} variant="h5">
            How is data quality ensured during input?
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Typography>
            The system includes validation checks, required field verification, and data format standardization to ensure consistent and accurate data entry across all users.
          </Typography>
        </AccordionDetails>
      </Accordion>

      <Accordion 
        expanded={expandedPanel === 'panel7'} 
        onChange={handleAccordionChange('panel7')}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography color={colors.greenAccent[500]} variant="h5">
            Can I edit or delete previously entered data?
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Typography>
            Yes, authorized users can edit or delete records through the symptoms management interface. All changes are logged for audit purposes and data integrity.
          </Typography>
        </AccordionDetails>
      </Accordion>

      <Typography
        mb="10px"
        mt="20px"
        fontWeight={600}
        variant="h4"
        color={colors.grey[100]}
      >
        Machine Learning Models and Training
      </Typography>
      <Accordion 
        expanded={expandedPanel === 'panel8'} 
        onChange={handleAccordionChange('panel8')}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography color={colors.greenAccent[500]} variant="h5">
            What machine learning models are used?
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Typography>
            The system uses XGBoost, Random Forest, Logistic Regression, and Naive Bayes models. These were selected for their accuracy, interpretability, and proven effectiveness in medical prediction tasks.
          </Typography>
        </AccordionDetails>
      </Accordion>

      <Accordion 
        expanded={expandedPanel === 'panel9'} 
        onChange={handleAccordionChange('panel9')}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography color={colors.greenAccent[500]} variant="h5">
            How are the models trained and updated?
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Typography>
            Models are trained on historical patient data together with environmental and academic calendar data. 
            They are periodically retrained as new data becomes available to ensure accuracy and relevance.
          </Typography>
        </AccordionDetails>
      </Accordion>
    </Box>
  );
};

export default FAQ;
