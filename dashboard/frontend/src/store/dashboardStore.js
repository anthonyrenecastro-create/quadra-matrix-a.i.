import { create } from 'zustand';
import axios from 'axios';

const API_BASE = '/api';
const WS_BASE = `ws://${window.location.hostname}:8000`;

export const useStore = create((set, get) => ({
  // Connection state
  connectionStatus: 'disconnected',
  websocket: null,
  
  // Cognition state
  cognitionData: null,
  isRunning: false,
  
  // Training state
  trainingMetrics: [],
  
  // Governance state
  policies: [],
  
  // Edge state
  edgeDevices: [],
  
  // Actions
  connectWebSocket: () => {
    const ws = new WebSocket(`${WS_BASE}/ws/cognition`);
    
    ws.onopen = () => {
      console.log('✅ WebSocket connected');
      set({ connectionStatus: 'connected', websocket: ws });
    };
    
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === 'cognition_update') {
        set({ cognitionData: message.data });
      }
    };
    
    ws.onerror = (error) => {
      console.error('❌ WebSocket error:', error);
      set({ connectionStatus: 'error' });
    };
    
    ws.onclose = () => {
      console.log('🔌 WebSocket disconnected');
      set({ connectionStatus: 'disconnected', websocket: null });
      
      // Attempt reconnect after 3 seconds
      setTimeout(() => {
        if (get().connectionStatus === 'disconnected') {
          get().connectWebSocket();
        }
      }, 3000);
    };
  },
  
  startCognition: async () => {
    try {
      await axios.post(`${API_BASE}/cognition/start`);
      set({ isRunning: true });
    } catch (error) {
      console.error('Failed to start cognition:', error);
    }
  },
  
  stopCognition: async () => {
    try {
      await axios.post(`${API_BASE}/cognition/stop`);
      set({ isRunning: false });
    } catch (error) {
      console.error('Failed to stop cognition:', error);
    }
  },
  
  sendInteraction: async (command) => {
    try {
      const response = await axios.post(`${API_BASE}/cognition/interact`, command);
      return response.data;
    } catch (error) {
      console.error('Interaction failed:', error);
      throw error;
    }
  },
  
  fetchPolicies: async () => {
    try {
      const response = await axios.get(`${API_BASE}/governance/policies`);
      set({ policies: response.data.policies });
    } catch (error) {
      console.error('Failed to fetch policies:', error);
    }
  },
  
  fetchEdgeDevices: async () => {
    try {
      const response = await axios.get(`${API_BASE}/edge/devices`);
      set({ edgeDevices: response.data.devices });
    } catch (error) {
      console.error('Failed to fetch edge devices:', error);
    }
  },
  
  fetchTrainingMetrics: async () => {
    try {
      const response = await axios.get(`${API_BASE}/training/metrics`);
      set({ trainingMetrics: response.data.metrics });
    } catch (error) {
      console.error('Failed to fetch training metrics:', error);
    }
  },
}));
