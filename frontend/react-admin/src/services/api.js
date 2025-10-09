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
    console.group('🚀 API Request Debug');
    console.log('URL:', config.url);
    console.log('Method:', config.method?.toUpperCase());
    console.log('Token exists:', !!token);
    console.log('Token preview:', token ? token.substring(0, 20) + '...' : 'No token');
    
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
      console.log('Authorization header set:', `Bearer ${token.substring(0, 20)}...`);
    } else {
      console.warn('⚠️ No token found - request will be unauthorized');
    }
    
    console.log('Request headers:', config.headers);
    console.log('Request data:', config.data);
    console.groupEnd();
    
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor to handle token refresh and error logging
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    // Enhanced error logging for debugging
    console.group('API Error Details:');
    console.log('URL:', error.config?.url);
    console.log('Method:', error.config?.method?.toUpperCase());
    console.log('Status:', error.response?.status);
    console.log('Status Text:', error.response?.statusText);
    console.log('Response Headers:', error.response?.headers);
    console.log('Response Data:', error.response?.data);
    console.log('Full Error:', error);
    console.groupEnd();

    // Handle server errors (5xx)
    if (error.response?.status >= 500) {
      console.error('Server Error:', {
        url: error.config?.url,
        status: error.response.status,
        data: error.response.data
      });
      
      // If response is HTML (server error page), create a more user-friendly error
      if (typeof error.response.data === 'string' && error.response.data.includes('<!doctype html>')) {
        const enhancedError = new Error(`Server Error (${error.response.status}): The backend server encountered an internal error.`);
        enhancedError.isServerError = true;
        enhancedError.originalStatus = error.response.status;
        enhancedError.originalUrl = error.config?.url;
        return Promise.reject(enhancedError);
      }
    }

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

// Debug and health check utilities
export const debugAPI = {
  // Test API connectivity
  healthCheck: async () => {
    try {
      console.log('Testing API health...');
      console.log('API Base URL:', API_BASE_URL);
      
      // Test basic connectivity
      const response = await axios.get(`${API_BASE_URL}/`, {
        timeout: 10000,
        headers: {
          'Accept': 'application/json',
        }
      });
      
      return {
        status: 'success',
        statusCode: response.status,
        data: response.data
      };
    } catch (error) {
      console.error('Health check failed:', error);
      return {
        status: 'error',
        error: error.message,
        statusCode: error.response?.status,
        data: error.response?.data
      };
    }
  },

  // Test specific endpoint
  testEndpoint: async (endpoint, method = 'GET', data = null) => {
    try {
      console.log(`Testing endpoint: ${method} ${endpoint}`);
      
      const config = {
        method: method.toLowerCase(),
        url: endpoint,
        timeout: 10000,
      };

      if (data && ['post', 'put', 'patch'].includes(method.toLowerCase())) {
        config.data = data;
      }

      const response = await api(config);
      
      console.log('✅ Endpoint test successful:', {
        url: endpoint,
        method,
        status: response.status,
        data: response.data
      });
      
      return {
        status: 'success',
        statusCode: response.status,
        data: response.data
      };
    } catch (error) {
      console.error('❌ Endpoint test failed:', {
        url: endpoint,
        method,
        error: error.message,
        status: error.response?.status,
        data: error.response?.data
      });
      
      return {
        status: 'error',
        error: error.message,
        statusCode: error.response?.status,
        data: error.response?.data
      };
    }
  },

  // Get current environment info
  getEnvironmentInfo: () => {
    return {
      apiUrl: API_BASE_URL,
      environment: process.env.NODE_ENV,
      reactAppApiUrl: process.env.REACT_APP_API_URL,
      hasToken: !!getToken(),
      userAgent: navigator.userAgent,
      timestamp: new Date().toISOString()
    };
  },

  // Log detailed request info for debugging
  logRequest: (config) => {
    console.group('🚀 API Request Debug');
    console.log('URL:', `${config.baseURL || API_BASE_URL}${config.url}`);
    console.log('Method:', config.method?.toUpperCase());
    console.log('Headers:', config.headers);
    console.log('Data:', config.data);
    console.log('Params:', config.params);
    console.groupEnd();
  }
};

export { setTokens, clearTokens, getToken };
export default api;
