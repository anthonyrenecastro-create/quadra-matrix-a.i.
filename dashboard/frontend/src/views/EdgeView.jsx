import React, { useEffect } from 'react';
import { useStore } from '../store/dashboardStore';

function EdgeView() {
  const { edgeDevices, fetchEdgeDevices } = useStore();
  
  useEffect(() => {
    fetchEdgeDevices();
    const interval = setInterval(fetchEdgeDevices, 10000);
    return () => clearInterval(interval);
  }, [fetchEdgeDevices]);
  
  const statusColor = (status) => {
    switch(status) {
      case 'online': return 'bg-green-500';
      case 'degraded': return 'bg-yellow-500';
      case 'offline': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };
  
  return (
    <div className="space-y-6">
      <div className="bg-neural-gray rounded-lg p-6 border border-gray-800">
        <h2 className="text-xl font-bold text-neural-green mb-2">
          📡 Edge Deployment Status
        </h2>
        <p className="text-sm text-gray-400">
          Monitor edge devices running distributed cognition inference
        </p>
      </div>
      
      {/* Device Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {edgeDevices.length === 0 ? (
          <div className="col-span-full p-12 text-center text-gray-500 border border-dashed border-gray-700 rounded-lg">
            <div className="text-4xl mb-4">📡</div>
            <p>No edge devices registered</p>
            <p className="text-sm mt-2">Deploy models to edge devices and they'll appear here</p>
          </div>
        ) : (
          edgeDevices.map((device) => (
            <div 
              key={device.device_id}
              className="bg-neural-gray rounded-lg p-5 border border-gray-800"
            >
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="font-bold text-white">{device.device_id}</h3>
                  <p className="text-xs text-gray-400 mt-1">v{device.model_version}</p>
                </div>
                <div className={`w-3 h-3 rounded-full ${statusColor(device.status)} animate-pulse`} />
              </div>
              
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Status:</span>
                  <span className={`font-bold ${
                    device.status === 'online' ? 'text-green-400' :
                    device.status === 'degraded' ? 'text-yellow-400' :
                    'text-red-400'
                  }`}>
                    {device.status.toUpperCase()}
                  </span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-gray-400">Inferences:</span>
                  <span className="text-white font-mono">{device.inference_count.toLocaleString()}</span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-gray-400">Latency:</span>
                  <span className="text-white font-mono">{device.avg_latency_ms.toFixed(1)}ms</span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-gray-400">Memory:</span>
                  <span className="text-white font-mono">{device.memory_usage_mb.toFixed(0)}MB</span>
                </div>
                
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Last heartbeat:</span>
                  <span className="text-gray-500">
                    {new Date(device.last_heartbeat * 1000).toLocaleTimeString()}
                  </span>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
      
      {/* Deployment Info */}
      <div className="bg-cyan-900/20 rounded-lg p-4 border border-cyan-700/50">
        <p className="text-sm text-cyan-300">
          <span className="font-bold">🚀 Edge Deployment:</span> Edge devices run lightweight inference
          models for low-latency cognition. Devices report metrics via heartbeat protocol.
        </p>
      </div>
    </div>
  );
}

export default EdgeView;
