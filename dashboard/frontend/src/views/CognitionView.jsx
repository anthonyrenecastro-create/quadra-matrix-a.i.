import React, { useEffect } from 'react';
import { useStore } from '../store/dashboardStore';
import CognitivePhysiologyPlane from '../components/CognitivePhysiologyPlane';
import IntentDeliberationPlane from '../components/IntentDeliberationPlane';
import WorldInterfacePlane from '../components/WorldInterfacePlane';
import InteractionPanel from '../components/InteractionPanel';

function CognitionView() {
  const { cognitionData, isRunning, startCognition, stopCognition } = useStore();

  return (
    <div className="space-y-6">
      {/* Control Bar */}
      <div className="bg-neural-gray rounded-lg p-4 border border-gray-800">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <h2 className="text-xl font-bold text-neural-green">
              Cognitive Architecture Visualization
            </h2>
            <span className="text-sm text-gray-400">
              Three-Plane Neural Command Center
            </span>
          </div>
          
          <div className="flex items-center space-x-4">
            <button
              onClick={isRunning ? stopCognition : startCognition}
              className={`px-6 py-2 rounded-lg font-bold transition-all ${
                isRunning
                  ? 'bg-red-600 hover:bg-red-700 text-white'
                  : 'bg-green-600 hover:bg-green-700 text-white'
              }`}
            >
              {isRunning ? '⏸ PAUSE' : '▶ ACTIVATE'}
            </button>
            
            {cognitionData && (
              <div className="text-sm text-gray-400">
                Step: {cognitionData.timestamp?.toFixed(2) || 'N/A'}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Three-Plane Layout */}
      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
        {/* Main Planes (3/4 width) */}
        <div className="xl:col-span-3 space-y-6">
          {/* Plane 1: Cognitive Physiology */}
          <div className="bg-neural-gray rounded-lg p-6 border-2 border-neural-green">
            <h3 className="text-lg font-bold text-neural-green mb-4">
              1. COGNITIVE PHYSIOLOGY
            </h3>
            <p className="text-sm text-gray-400 mb-4">
              Heart monitor of intelligence • Pulses • Flows • Pressure • Heat
            </p>
            <CognitivePhysiologyPlane data={cognitionData} />
          </div>

          {/* Plane 2: Intent & Deliberation */}
          <div className="bg-neural-gray rounded-lg p-6 border-2 border-neural-orange">
            <h3 className="text-lg font-bold text-neural-orange mb-4">
              2. INTENT & DELIBERATION
            </h3>
            <p className="text-sm text-gray-400 mb-4">
              Goals forming • Competing hypotheses • Collaborative interaction
            </p>
            <IntentDeliberationPlane data={cognitionData} />
          </div>

          {/* Plane 3: World Interface */}
          <div className="bg-neural-gray rounded-lg p-6 border-2 border-neural-pink">
            <h3 className="text-lg font-bold text-neural-pink mb-4">
              3. WORLD INTERFACE
            </h3>
            <p className="text-sm text-gray-400 mb-4">
              Sensors • Proposed actions • External coupling
            </p>
            <WorldInterfacePlane data={cognitionData} />
          </div>
        </div>

        {/* Interaction Panel (1/4 width) */}
        <div className="xl:col-span-1">
          <InteractionPanel />
        </div>
      </div>

      {/* Living System Indicator */}
      {cognitionData && (
        <div className="bg-neural-gray rounded-lg p-4 border border-gray-800 text-center">
          <p className="text-neural-green italic">
            "If frozen for 10 seconds, it still feels alive"
          </p>
          <div className="mt-2 flex justify-center space-x-6 text-sm text-gray-400">
            <span>Coherence: {(cognitionData.field_coherence * 100).toFixed(1)}%</span>
            <span>Memory: {(cognitionData.memory_magnitude * 100).toFixed(1)}%</span>
            <span>Active Goals: {cognitionData.goals?.length || 0}</span>
          </div>
        </div>
      )}
    </div>
  );
}

export default CognitionView;
