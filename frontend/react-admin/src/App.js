import { useState } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import Topbar from "./scenes/global/topbar";
import Sidebar from "./scenes/global/sidebar";
import Dashboard from "./scenes/dashboard";
import Team from "./scenes/team";
import Invoices from "./scenes/symptoms";
import Contacts from "./scenes/contacts";
import Bar from "./scenes/bar";
import Form from "./scenes/form";
import Line from "./scenes/line";
import Pie from "./scenes/pie";
import FAQ from "./scenes/faq";
import Geography from "./scenes/geography";
import Heatmap from "./scenes/heatmap";
import { CssBaseline, ThemeProvider } from "@mui/material";
import { ColorModeContext, useMode } from "./theme";
import Calendar from "./scenes/calendar/calendar";  
import Symptoms from "./scenes/symptoms";
// import SymptomsForm from "./scenes/symptoms/SymptomsForm";
// import SymptomsView from "./scenes/symptoms/SymptomsView";
import AuthForm from "./scenes/auth/AuthForm";
import ModelTraining from "./scenes/modeltraining";
import DataPreprocessing from "./scenes/datapreprocessing";
import Help from "./scenes/help";
import Profile from "./scenes/profile";

// Protected Route Component
const ProtectedRoute = ({ children }) => {
  const token = localStorage.getItem('token');
  return token ? children : <Navigate to="/login" replace />;
};

function App() {
  const [theme, colorMode] = useMode();
  const [isSidebar, setIsSidebar] = useState(true);

  return (
    <ColorModeContext.Provider value={colorMode}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <div className="app">
          <Routes>
            {/* Default redirect to login */}
            <Route path="/" element={<Navigate to="/login" replace />} />
            
            {/* Authentication Routes */}
            <Route path="/login" element={<AuthForm />} />
            <Route path="/register" element={<AuthForm />} />
            
            {/* Protected Main App Routes */}
            <Route path="/dashboard" element={
              <ProtectedRoute>
                <Sidebar isSidebar={isSidebar} setIsSidebar={setIsSidebar} />
                <main className={`content ${!isSidebar ? 'collapsed' : ''}`}>
                  <Topbar setIsSidebar={setIsSidebar} />
                  <Dashboard />
                </main>
              </ProtectedRoute>
            } />
            
            <Route path="/app/*" element={
              <ProtectedRoute>
                <Sidebar isSidebar={isSidebar} setIsSidebar={setIsSidebar} />
                <main className={`content ${!isSidebar ? 'collapsed' : ''}`}>
                  <Topbar setIsSidebar={setIsSidebar} />
                  <Routes>
                    <Route path="/" element={<Navigate to="/dashboard" replace />} />
                    <Route path="/team" element={<Team />} />
                    <Route path="/contacts" element={<Contacts />} />
                    <Route path="/invoices" element={<Invoices />} />
                    <Route path="/form" element={<Form />} />
                    <Route path="/bar" element={<Bar />} />
                    <Route path="/pie" element={<Pie />} />
                    <Route path="/line" element={<Line />} />
                    <Route path="/faq" element={<FAQ />} />
                    <Route path="/calendar" element={<Calendar />} />
                    <Route path="/geography" element={<Geography />} />
                    <Route path="/symptoms" element={<Symptoms />} />
                    <Route path="/symptoms/create" element={<Form />} />
                    <Route path="/symptoms/edit/:id" element={<Form />} />
                    {/* <Route path="/symptoms/view/:id" element={<SymptomsView />} /> */}
                    <Route path="/modeltraining" element={<ModelTraining />} />
                    <Route path="/datapreprocessing" element={<DataPreprocessing />} />
                    <Route path="/help" element={<Help />} />
                    <Route path="/profile" element={<Profile />} />
                    <Route path="/heatmap" element={<Heatmap />} />
                  </Routes>
                </main>
              </ProtectedRoute>
            } />
          </Routes>
        </div>
      </ThemeProvider>
    </ColorModeContext.Provider>
  );
}

export default App;