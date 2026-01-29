import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 seconds timeout
});

/**
 * Predict emotion from text
 * @param {string} text - The text to analyze
 * @returns {Promise<Object>} - Prediction results
 */
export const predictEmotion = async (text) => {
  try {
    const response = await api.post('/predict', { text });
    return response.data;
  } catch (error) {
    if (error.response) {
      // Server responded with error
      throw new Error(error.response.data.detail || 'Failed to analyze emotion');
    } else if (error.request) {
      // Request made but no response
      throw new Error('Cannot connect to API server. Please ensure the backend is running.');
    } else {
      // Something else happened
      throw new Error('An unexpected error occurred');
    }
  }
};

/**
 * Get health status of the API
 * @returns {Promise<Object>} - Health check results
 */
export const getHealthCheck = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    throw new Error('Failed to check API health');
  }
};

/**
 * Get list of supported emotions
 * @returns {Promise<Object>} - List of emotions
 */
export const getEmotions = async () => {
  try {
    const response = await api.get('/emotions');
    return response.data;
  } catch (error) {
    throw new Error('Failed to fetch emotions list');
  }
};

/**
 * Get API information
 * @returns {Promise<Object>} - API info
 */
export const getApiInfo = async () => {
  try {
    const response = await api.get('/');
    return response.data;
  } catch (error) {
    throw new Error('Failed to fetch API information');
  }
};

export default api;
