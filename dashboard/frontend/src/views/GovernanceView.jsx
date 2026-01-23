import React, { useEffect } from 'react';
import { useStore } from '../store/dashboardStore';

function GovernanceView() {
  const { policies, fetchPolicies } = useStore();
  
  useEffect(() => {
    fetchPolicies();
  }, [fetchPolicies]);
  
  return (
    <div className="space-y-6">
      <div className="bg-neural-gray rounded-lg p-6 border border-gray-800">
        <h2 className="text-xl font-bold text-neural-green mb-2">
          ⚖️ Governance & Policy Management
        </h2>
        <p className="text-sm text-gray-400">
          Configure safety boundaries, ethical constraints, and compliance policies
        </p>
      </div>
      
      {/* Policy List */}
      <div className="space-y-4">
        {policies.map((policy) => (
          <div 
            key={policy.policy_id}
            className="bg-neural-gray rounded-lg p-6 border border-gray-800"
          >
            <div className="flex items-start justify-between mb-4">
              <div className="flex-1">
                <div className="flex items-center space-x-3 mb-2">
                  <h3 className="text-lg font-bold text-white">{policy.name}</h3>
                  <span className={`px-3 py-1 rounded text-xs font-bold ${
                    policy.type === 'hard_boundary' ? 'bg-red-900 text-red-300' :
                    policy.type === 'soft_penalty' ? 'bg-yellow-900 text-yellow-300' :
                    'bg-blue-900 text-blue-300'
                  }`}>
                    {policy.type.replace('_', ' ').toUpperCase()}
                  </span>
                </div>
                <p className="text-sm text-gray-400">{policy.description}</p>
              </div>
              
              <label className="flex items-center space-x-2 cursor-pointer">
                <input 
                  type="checkbox" 
                  checked={policy.enabled}
                  onChange={() => {/* TODO: Update policy */}}
                  className="w-5 h-5 rounded bg-neutral-900 border-gray-700"
                />
                <span className="text-sm text-gray-400">Enabled</span>
              </label>
            </div>
            
            {/* Parameters */}
            <div className="bg-neutral-900 rounded p-4 border border-gray-800">
              <div className="text-xs font-mono text-gray-400">
                <div className="font-bold text-white mb-2">Parameters:</div>
                <pre className="text-gray-400">
                  {JSON.stringify(policy.parameters, null, 2)}
                </pre>
              </div>
            </div>
          </div>
        ))}
      </div>
      
      {/* Add New Policy */}
      <button className="w-full px-4 py-3 bg-green-700 hover:bg-green-600 text-white rounded-lg font-bold transition-colors">
        + Add New Policy
      </button>
      
      {/* Compliance Notice */}
      <div className="bg-purple-900/20 rounded-lg p-4 border border-purple-700/50">
        <p className="text-sm text-purple-300">
          <span className="font-bold">📋 Compliance:</span> All policies are logged and auditable. 
          Hard boundaries cannot be overridden. Soft penalties can be adjusted dynamically.
        </p>
      </div>
    </div>
  );
}

export default GovernanceView;
