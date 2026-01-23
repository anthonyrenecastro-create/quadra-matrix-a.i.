import React, { useEffect } from 'react';
import { useStore } from '../store/dashboardStore';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

function TrainingView() {
  const { trainingMetrics, fetchTrainingMetrics } = useStore();
  
  useEffect(() => {
    fetchTrainingMetrics();
    const interval = setInterval(fetchTrainingMetrics, 5000);
    return () => clearInterval(interval);
  }, [fetchTrainingMetrics]);
  
  const latestMetric = trainingMetrics[trainingMetrics.length - 1];
  
  return (
    <div className="space-y-6">
      <div className="bg-neural-gray rounded-lg p-6 border border-gray-800">
        <h2 className="text-xl font-bold text-neural-green mb-2">
          📊 Training Dashboard
        </h2>
        <p className="text-sm text-gray-400">
          Live training metrics and model performance monitoring
        </p>
      </div>
      
      {/* Metrics Cards */}
      {latestMetric && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <MetricCard title="Loss" value={latestMetric.loss?.toFixed(4)} color="text-red-400" />
          <MetricCard title="Perplexity" value={latestMetric.perplexity?.toFixed(2)} color="text-yellow-400" />
          <MetricCard title="Learning Rate" value={latestMetric.learning_rate?.toExponential(2)} color="text-blue-400" />
          <MetricCard title="Gradient Norm" value={latestMetric.gradient_norm?.toFixed(3)} color="text-purple-400" />
        </div>
      )}
      
      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartCard title="Loss Over Time">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={trainingMetrics}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="step" stroke="#888" />
              <YAxis stroke="#888" />
              <Tooltip contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333' }} />
              <Line type="monotone" dataKey="loss" stroke="#ff4444" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
        
        <ChartCard title="Learning Rate">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={trainingMetrics}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="step" stroke="#888" />
              <YAxis stroke="#888" />
              <Tooltip contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333' }} />
              <Line type="monotone" dataKey="learning_rate" stroke="#4488ff" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
      
      {/* Integration Notice */}
      <div className="bg-blue-900/20 rounded-lg p-4 border border-blue-700/50">
        <p className="text-sm text-blue-300">
          <span className="font-bold">🔗 Integration:</span> This view connects to train_live_dashboard.py 
          via the /api/training/metrics endpoint. Training data streams in real-time via WebSocket.
        </p>
      </div>
    </div>
  );
}

function MetricCard({ title, value, color }) {
  return (
    <div className="bg-neural-gray rounded-lg p-4 border border-gray-800">
      <div className="text-sm text-gray-400 mb-1">{title}</div>
      <div className={`text-2xl font-bold ${color}`}>
        {value || 'N/A'}
      </div>
    </div>
  );
}

function ChartCard({ title, children }) {
  return (
    <div className="bg-neural-gray rounded-lg p-4 border border-gray-800">
      <h3 className="text-sm font-bold text-white mb-4">{title}</h3>
      {children}
    </div>
  );
}

export default TrainingView;
