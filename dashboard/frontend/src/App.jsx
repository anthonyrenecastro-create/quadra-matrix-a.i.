import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import CognitionView from './views/CognitionView';
import TrainingView from './views/TrainingView';
import GovernanceView from './views/GovernanceView';
import EdgeView from './views/EdgeView';
import { useStore } from './store/dashboardStore';
import './styles/App.css';

function App() {
  const { connectionStatus, connectWebSocket } = useStore();
  const [activeTab, setActiveTab] = useState('cognition');

  console.log('App rendering, connectionStatus:', connectionStatus);

  useEffect(() => {
    // Connect to WebSocket on mount
    connectWebSocket();
  }, [connectWebSocket]);

  const tabs = [
    { id: 'cognition', name: 'Cognition', icon: '🧠', path: '/' },
    { id: 'training', name: 'Training', icon: '📊', path: '/training' },
    { id: 'governance', name: 'Governance', icon: '⚖️', path: '/governance' },
    { id: 'edge', name: 'Edge Devices', icon: '📡', path: '/edge' },
  ];

  return (
    <Router>
      <div className="min-h-screen bg-neural-dark text-white">
        {/* Header */}
        <header className="bg-neural-gray border-b-2 border-neural-green">
          <div className="max-w-full px-6 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <h1 className="text-2xl font-bold text-neural-green">
                  ⚡ NEURAL COMMAND CENTER
                </h1>
                <span className="text-sm text-gray-400">
                  CognitionSim Cognitive Architecture
                </span>
              </div>
              
              <div className="flex items-center space-x-4">
                {/* Connection Status */}
                <div className="flex items-center space-x-2">
                  <div className={`w-3 h-3 rounded-full ${
                    connectionStatus === 'connected' ? 'bg-green-500 animate-pulse' :
                    connectionStatus === 'connecting' ? 'bg-yellow-500' :
                    'bg-red-500'
                  }`} />
                  <span className="text-sm text-gray-400">
                    {connectionStatus}
                  </span>
                </div>
                
                {/* Version */}
                <span className="text-xs text-gray-500 px-3 py-1 bg-neural-gray rounded">
                  v2.0.0
                </span>
              </div>
            </div>

            {/* Navigation Tabs */}
            <nav className="mt-4 flex space-x-2">
              {tabs.map((tab) => (
                <Link
                  key={tab.id}
                  to={tab.path}
                  onClick={() => setActiveTab(tab.id)}
                  className={`px-4 py-2 rounded-t-lg transition-colors ${
                    activeTab === tab.id
                      ? 'bg-neural-dark text-neural-green border-t-2 border-x-2 border-neural-green'
                      : 'bg-neural-gray text-gray-400 hover:text-white hover:bg-neutral-800'
                  }`}
                >
                  <span className="mr-2">{tab.icon}</span>
                  {tab.name}
                </Link>
              ))}
            </nav>
          </div>
        </header>

        {/* Main Content */}
        <main className="max-w-full p-6">
          <Routes>
            <Route path="/" element={<CognitionView />} />
            <Route path="/training" element={<TrainingView />} />
            <Route path="/governance" element={<GovernanceView />} />
            <Route path="/edge" element={<EdgeView />} />
          </Routes>
        </main>

        {/* Footer */}
        <footer className="bg-neural-gray border-t border-gray-800 py-4 text-center text-gray-500 text-sm">
          <p>Not a Chatbot • A Living Cognitive System • {new Date().getFullYear()}</p>
        </footer>
      </div>
    </Router>
  );
}

export default App;
