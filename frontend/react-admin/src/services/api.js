import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Token management
const getToken = () => localStorage.getItem('token');
const getRefreshToken = () => localStorage.getItem('refresh_token');
const setTokens = (token, refresh) => {
  localStorage.setItem('token', token);
  if (refresh) localStorage.setItem('refresh_token', refresh);
};
const clearTokens = () => {
  localStorage.removeItem('token');
  localStorage.removeItem('refresh_token');
  localStorage.removeItem('user');
};

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = getToken();
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor to handle token refresh
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;

      try {
        const refreshToken = getRefreshToken();
        if (refreshToken) {
          const response = await axios.post(`${API_BASE_URL}/api/auth/refresh/`, {
            refresh: refreshToken,
          });
          
          const { access } = response.data;
          setTokens(access, refreshToken);
          
          // Retry original request with new token
          originalRequest.headers.Authorization = `Bearer ${access}`;
          return api(originalRequest);
        }
      } catch (refreshError) {
        clearTokens();
        window.location.href = '/login';
      }
    }

    return Promise.reject(error);
  }
);

// Auth API
export const authAPI = {
  login: (credentials) => api.post('/api/auth/login/', credentials),
  register: (userData) => api.post('/api/auth/register/', userData),
  logout: (refreshToken) => api.post('/api/auth/logout/', { refresh: refreshToken }),
  refreshToken: (refreshToken) => api.post('/api/auth/refresh/', { refresh: refreshToken }),
  getProfile: () => api.get('/api/auth/profile/'),
  updateProfile: (data) => api.patch('/api/auth/profile/', data),
};

// Dashboard API
export const dashboardAPI = {
  getStats: () => api.get('/api/auth/dashboard/stats/'),
};

// Patients API
export const patientsAPI = {
  getAll: (params) => api.get('/api/patients/patients/', { params }),
  getById: (id) => api.get(`/api/patients/patients/${id}/`),
  create: (data) => api.post('/api/patients/patients/', data),
  update: (id, data) => api.patch(`/api/patients/patients/${id}/`, data),
  delete: (id) => api.delete(`/api/patients/patients/${id}/`),
  getStatistics: () => api.get('/api/patients/patients/statistics/'),
  getSymptoms: (id) => api.get(`/api/patients/patients/${id}/symptoms/`),
};

// Symptoms API
export const symptomsAPI = {
  getAll: (params) => api.get('/api/patients/symptoms/', { params }),
  getById: (id) => api.get(`/api/patients/symptoms/${id}/`),
  create: (data) => api.post('/api/patients/symptoms/', data),
  update: (id, data) => api.patch(`/api/patients/symptoms/${id}/`, data),
  delete: (id) => api.delete(`/api/patients/symptoms/${id}/`),
  getCommonSymptoms: () => api.get('/api/patients/symptoms/common_symptoms/'),
  getTrends: () => api.get('/api/patients/symptoms/symptom_trends/'),
};

// Activity Logs API
export const activityAPI = {
  getAll: (params) => api.get('/api/auth/activity-logs/', { params }),
  getById: (id) => api.get(`/api/auth/activity-logs/${id}/`),
  create: (data) => api.post('/api/auth/activity-logs/', data),
  update: (id, data) => api.patch(`/api/auth/activity-logs/${id}/`, data),
  delete: (id) => api.delete(`/api/auth/activity-logs/${id}/`),
};

export { setTokens, clearTokens, getToken };
export default api;
