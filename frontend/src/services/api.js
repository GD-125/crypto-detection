const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

class ApiService {
  async request(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    const config = {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    };

    try {
      const response = await fetch(url, config);

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'API request failed');
      }

      return await response.json();
    } catch (error) {
      console.error('API Error:', error);
      throw error;
    }
  }

  // Dashboard
  async getDashboardStats() {
    return this.request('/dashboard/stats');
  }

  async getRecentActivity() {
    return this.request('/dashboard/recent-activity');
  }

  // Firmware
  async uploadFirmware(file, architecture = 'auto') {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/firmware/upload?architecture=${architecture}`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Upload failed');
    }

    return await response.json();
  }

  async getFirmware(id) {
    return this.request(`/firmware/${id}`);
  }

  async listFirmware(skip = 0, limit = 100) {
    return this.request(`/firmware/list?skip=${skip}&limit=${limit}`);
  }

  async deleteFirmware(id) {
    return this.request(`/firmware/${id}`, { method: 'DELETE' });
  }

  // Analysis
  async startAnalysis(firmwareId, options = {}) {
    return this.request(`/analysis/start/${firmwareId}`, {
      method: 'POST',
      body: JSON.stringify(options),
    });
  }

  async getAnalysisStatus(firmwareId) {
    return this.request(`/analysis/status/${firmwareId}`);
  }

  async startBatchAnalysis(firmwareIds) {
    return this.request('/analysis/batch', {
      method: 'POST',
      body: JSON.stringify(firmwareIds),
    });
  }

  // Results
  async getResults(firmwareId) {
    return this.request(`/results/${firmwareId}`);
  }

  async listAllResults(skip = 0, limit = 100) {
    return this.request(`/results/list/all?skip=${skip}&limit=${limit}`);
  }

  async exportResults(firmwareId, format = 'json') {
    return this.request(`/results/export/${firmwareId}?format=${format}`);
  }
}

const apiService = new ApiService();
export default apiService;
