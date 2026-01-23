import React from 'react';
import { useStore } from '../store/dashboardStore';

function WorldInterfacePlane({ data }) {
  const { sendInteraction } = useStore();
  const sensors = data?.sensors || { visual: 0, temporal: 0, pattern: 0 };
  const actions = data?.proposed_actions || [];
  
  const handleApproveAction = (actionName) => {
    sendInteraction({
      command_type: 'approve_action',
      target: actionName
    });
  };
  
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Left: Sensors */}
      <div>
        <h4 className="text-sm font-bold text-neural-pink mb-4">SENSOR INPUTS</h4>
        <div className="space-y-4">
          {Object.entries(sensors).map(([sensorName, value]) => (
            <div key={sensorName} className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm font-mono uppercase text-gray-300">
                  {sensorName}
                </span>
                <span className="text-sm font-mono text-neural-cyan">
                  {(value * 100).toFixed(1)}%
                </span>
              </div>
              
              {/* Progress bar */}
              <div className="h-8 bg-neutral-900 rounded-lg overflow-hidden border border-gray-700">
                <div 
                  className="h-full bg-gradient-to-r from-neural-cyan to-cyan-300 transition-all duration-300"
                  style={{ width: `${value * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
        
        {/* Sensor Legend */}
        <div className="mt-6 p-3 bg-neutral-900 rounded border border-gray-700 text-xs text-gray-400 space-y-1">
          <div><span className="text-neural-cyan">Visual:</span> Spatial input strength</div>
          <div><span className="text-neural-cyan">Temporal:</span> Time-series patterns</div>
          <div><span className="text-neural-cyan">Pattern:</span> Coherence detection</div>
        </div>
      </div>
      
      {/* Right: Proposed Actions */}
      <div>
        <h4 className="text-sm font-bold text-neural-pink mb-4">PROPOSED ACTIONS</h4>
        <div className="space-y-3">
          {actions.length === 0 ? (
            <div className="p-6 text-center text-gray-500 border border-dashed border-gray-700 rounded-lg">
              <p className="text-sm">No actions proposed yet</p>
              <p className="text-xs mt-2">System will generate actions based on current state</p>
            </div>
          ) : (
            actions.slice(0, 5).map((action, idx) => (
              <div 
                key={idx}
                className={`p-4 rounded-lg border transition-all ${
                  action.approved 
                    ? 'bg-green-900/20 border-green-600' 
                    : 'bg-neutral-900 border-gray-700 hover:border-neural-pink'
                }`}
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2">
                      <span className={`text-sm font-bold ${
                        action.approved ? 'text-green-400' : 'text-yellow-400'
                      }`}>
                        {action.approved ? '✓' : '→'} {action.name}
                      </span>
                      {action.uncertainty > 0.3 && (
                        <span className="text-xs px-2 py-0.5 bg-red-900/40 text-red-400 rounded">
                          ±{(action.uncertainty * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-gray-400 mt-1">{action.outcome_estimate}</p>
                  </div>
                  
                  {!action.approved && (
                    <button
                      onClick={() => handleApproveAction(action.name)}
                      className="ml-2 px-3 py-1 text-xs bg-green-700 hover:bg-green-600 text-white rounded transition-colors"
                    >
                      Approve
                    </button>
                  )}
                </div>
                
                {/* Magnitude bar */}
                <div className="h-2 bg-neutral-800 rounded overflow-hidden">
                  <div 
                    className={`h-full transition-all ${
                      action.approved ? 'bg-green-500' : 'bg-yellow-500'
                    }`}
                    style={{ width: `${action.magnitude * 100}%` }}
                  />
                </div>
              </div>
            ))
          )}
        </div>
        
        {/* Action Policy Notice */}
        <div className="mt-6 p-3 bg-yellow-900/20 rounded border border-yellow-700/50 text-xs text-yellow-400">
          <p className="font-bold">⚠️ Safety Policy</p>
          <p className="mt-1">System never acts directly. All actions require human approval or sandbox testing.</p>
        </div>
      </div>
    </div>
  );
}

export default WorldInterfacePlane;
