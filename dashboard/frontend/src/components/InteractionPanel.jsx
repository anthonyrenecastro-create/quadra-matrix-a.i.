import React, { useState } from 'react';
import { useStore } from '../store/dashboardStore';

function InteractionPanel() {
  const { sendInteraction } = useStore();
  const [selectedSensor, setSelectedSensor] = useState('pattern');
  const [constraintText, setConstraintText] = useState('');
  
  const handleInjectPulse = () => {
    sendInteraction({
      command_type: 'inject_signal',
      target: selectedSensor,
      parameters: { intensity: 1.0 }
    });
  };
  
  const handleSetConstraint = () => {
    if (constraintText.trim()) {
      sendInteraction({
        command_type: 'inject_constraint',
        target: constraintText
      });
      setConstraintText('');
    }
  };
  
  const handleProbe = (probeType) => {
    sendInteraction({
      command_type: 'probe',
      target: probeType
    });
  };
  
  return (
    <div className="space-y-4 sticky top-6">
      <div className="bg-neural-gray rounded-lg p-4 border border-neural-green">
        <h3 className="text-lg font-bold text-neural-green mb-4">
          ⚡ INTERACTION MODES
        </h3>
        
        {/* Signal Injection */}
        <div className="mb-6">
          <h4 className="text-sm font-bold text-white mb-3">Signal Injection</h4>
          <div className="space-y-2">
            <select 
              value={selectedSensor}
              onChange={(e) => setSelectedSensor(e.target.value)}
              className="w-full px-3 py-2 bg-neutral-900 text-white rounded border border-gray-700 text-sm"
            >
              <option value="visual">Visual Stream</option>
              <option value="temporal">Temporal Stream</option>
              <option value="pattern">Pattern Stream</option>
            </select>
            
            <button
              onClick={handleInjectPulse}
              className="w-full px-4 py-2 bg-blue-700 hover:bg-blue-600 text-white rounded text-sm font-bold transition-colors"
            >
              💉 Inject Data Pulse
            </button>
          </div>
        </div>
        
        {/* Cognitive Probes */}
        <div className="mb-6">
          <h4 className="text-sm font-bold text-white mb-3">Cognitive Probes</h4>
          <div className="space-y-2">
            <button
              onClick={() => handleProbe('uncertainty')}
              className="w-full px-3 py-2 bg-purple-800 hover:bg-purple-700 text-white rounded text-xs transition-colors"
            >
              🔍 Show Uncertainty
            </button>
            <button
              onClick={() => handleProbe('conflicts')}
              className="w-full px-3 py-2 bg-purple-800 hover:bg-purple-700 text-white rounded text-xs transition-colors"
            >
              ⚠️ Find Conflicts
            </button>
            <button
              onClick={() => handleProbe('stability')}
              className="w-full px-3 py-2 bg-purple-800 hover:bg-purple-700 text-white rounded text-xs transition-colors"
            >
              📊 Test Stability
            </button>
          </div>
        </div>
        
        {/* Safety Governors */}
        <div className="mb-6">
          <h4 className="text-sm font-bold text-white mb-3">Safety Governors</h4>
          <div className="space-y-2">
            <input
              type="text"
              value={constraintText}
              onChange={(e) => setConstraintText(e.target.value)}
              placeholder="Enter constraint..."
              className="w-full px-3 py-2 bg-neutral-900 text-white rounded border border-gray-700 text-xs"
            />
            <button
              onClick={handleSetConstraint}
              className="w-full px-3 py-2 bg-red-700 hover:bg-red-600 text-white rounded text-xs font-bold transition-colors"
            >
              🛡️ Set Constraint
            </button>
          </div>
        </div>
        
        {/* Quick Stats */}
        <div className="p-3 bg-neutral-900 rounded border border-gray-700">
          <h4 className="text-xs font-bold text-gray-400 mb-2">SYSTEM STATUS</h4>
          <div className="space-y-1 text-xs text-gray-500">
            <div className="flex justify-between">
              <span>Mode:</span>
              <span className="text-neural-green">Autonomous</span>
            </div>
            <div className="flex justify-between">
              <span>Safety:</span>
              <span className="text-green-400">Active</span>
            </div>
            <div className="flex justify-between">
              <span>Oversight:</span>
              <span className="text-yellow-400">Human-in-loop</span>
            </div>
          </div>
        </div>
      </div>
      
      {/* Philosophy Note */}
      <div className="bg-neural-gray rounded-lg p-4 border border-gray-700">
        <p className="text-xs text-gray-400 italic">
          "No chat box. You don't command it. You collaborate with it through observation and influence."
        </p>
      </div>
    </div>
  );
}

export default InteractionPanel;
