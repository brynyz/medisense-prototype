import { useQuery, useMutation, useQueryClient } from 'react-query';
import { toast } from 'react-toastify';
import { 
  dashboardAPI, 
  patientsAPI, 
  symptomsAPI, 
  activityAPI 
} from '../services/api';

// Dashboard hooks
export const useDashboardStats = () => {
  return useQuery('dashboardStats', dashboardAPI.getStats, {
    staleTime: 5 * 60 * 1000, // 5 minutes
    cacheTime: 10 * 60 * 1000, // 10 minutes
  });
};

// Patient hooks
export const usePatients = (params = {}) => {
  return useQuery(['patients', params], () => patientsAPI.getAll(params), {
    keepPreviousData: true,
  });
};

export const usePatient = (id) => {
  return useQuery(['patient', id], () => patientsAPI.getById(id), {
    enabled: !!id,
  });
};

export const usePatientStatistics = () => {
  return useQuery('patientStatistics', patientsAPI.getStatistics, {
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
};

export const useCreatePatient = () => {
  const queryClient = useQueryClient();
  
  return useMutation(patientsAPI.create, {
    onSuccess: () => {
      queryClient.invalidateQueries('patients');
      queryClient.invalidateQueries('patientStatistics');
      toast.success('Patient created successfully');
    },
    onError: (error) => {
      const message = error.response?.data?.detail || 'Failed to create patient';
      toast.error(message);
    },
  });
};

export const useUpdatePatient = () => {
  const queryClient = useQueryClient();
  
  return useMutation(
    ({ id, data }) => patientsAPI.update(id, data),
    {
      onSuccess: (data, variables) => {
        queryClient.invalidateQueries('patients');
        queryClient.invalidateQueries(['patient', variables.id]);
        queryClient.invalidateQueries('patientStatistics');
        toast.success('Patient updated successfully');
      },
      onError: (error) => {
        const message = error.response?.data?.detail || 'Failed to update patient';
        toast.error(message);
      },
    }
  );
};

export const useDeletePatient = () => {
  const queryClient = useQueryClient();
  
  return useMutation(patientsAPI.delete, {
    onSuccess: () => {
      queryClient.invalidateQueries('patients');
      queryClient.invalidateQueries('patientStatistics');
      toast.success('Patient deleted successfully');
    },
    onError: (error) => {
      const message = error.response?.data?.detail || 'Failed to delete patient';
      toast.error(message);
    },
  });
};

// Symptom hooks
export const useSymptoms = (params = {}) => {
  return useQuery(['symptoms', params], () => symptomsAPI.getAll(params), {
    keepPreviousData: true,
  });
};

export const useSymptom = (id) => {
  return useQuery(['symptom', id], () => symptomsAPI.getById(id), {
    enabled: !!id,
  });
};

export const useCommonSymptoms = () => {
  return useQuery('commonSymptoms', symptomsAPI.getCommonSymptoms, {
    staleTime: 15 * 60 * 1000, // 15 minutes
  });
};

export const useSymptomTrends = () => {
  return useQuery('symptomTrends', symptomsAPI.getTrends, {
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
};

export const useCreateSymptom = () => {
  const queryClient = useQueryClient();
  
  return useMutation(symptomsAPI.create, {
    onSuccess: () => {
      queryClient.invalidateQueries('symptoms');
      queryClient.invalidateQueries('commonSymptoms');
      queryClient.invalidateQueries('symptomTrends');
      toast.success('Symptom log created successfully');
    },
    onError: (error) => {
      const message = error.response?.data?.detail || 'Failed to create symptom log';
      toast.error(message);
    },
  });
};

export const useUpdateSymptom = () => {
  const queryClient = useQueryClient();
  
  return useMutation(
    ({ id, data }) => symptomsAPI.update(id, data),
    {
      onSuccess: (data, variables) => {
        queryClient.invalidateQueries('symptoms');
        queryClient.invalidateQueries(['symptom', variables.id]);
        queryClient.invalidateQueries('commonSymptoms');
        queryClient.invalidateQueries('symptomTrends');
        toast.success('Symptom log updated successfully');
      },
      onError: (error) => {
        const message = error.response?.data?.detail || 'Failed to update symptom log';
        toast.error(message);
      },
    }
  );
};

export const useDeleteSymptom = () => {
  const queryClient = useQueryClient();
  
  return useMutation(symptomsAPI.delete, {
    onSuccess: () => {
      queryClient.invalidateQueries('symptoms');
      queryClient.invalidateQueries('commonSymptoms');
      queryClient.invalidateQueries('symptomTrends');
      toast.success('Symptom log deleted successfully');
    },
    onError: (error) => {
      const message = error.response?.data?.detail || 'Failed to delete symptom log';
      toast.error(message);
    },
  });
};

// Activity log hooks
export const useActivityLogs = (params = {}) => {
  return useQuery(['activityLogs', params], () => activityAPI.getAll(params), {
    keepPreviousData: true,
  });
};
