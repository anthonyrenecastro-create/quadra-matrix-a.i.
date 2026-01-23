import React from 'react';
import ReactDOM from 'react-dom/client';
import './styles/index.css';

function TestApp() {
  return (
    <div style={{ 
      background: '#111', 
      color: '#0f0', 
      padding: '50px', 
      fontSize: '24px',
      minHeight: '100vh'
    }}>
      <h1>🚀 Neural Command Center - TEST MODE</h1>
      <p>If you see this, React is working!</p>
      <p>Time: {new Date().toLocaleTimeString()}</p>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <TestApp />
  </React.StrictMode>
);
