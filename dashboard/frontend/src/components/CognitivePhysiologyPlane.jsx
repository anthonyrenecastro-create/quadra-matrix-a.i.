import React, { useEffect, useRef } from 'react';

function CognitivePhysiologyPlane({ data }) {
  const canvasRef = useRef(null);
  
  useEffect(() => {
    if (!data || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, width, height);
    
    const layers = data.layers || {};
    const layerPositions = {
      perception: [width * 0.25, height * 0.25],
      integration: [width * 0.75, height * 0.25],
      reasoning: [width * 0.25, height * 0.75],
      action: [width * 0.75, height * 0.75]
    };
    
    const radius = Math.min(width, height) * 0.15;
    
    // Draw connections (flows)
    const connections = [
      ['perception', 'integration'],
      ['integration', 'reasoning'],
      ['reasoning', 'action'],
      ['perception', 'reasoning'],
      ['integration', 'action']
    ];
    
    connections.forEach(([layer1, layer2]) => {
      const [x1, y1] = layerPositions[layer1];
      const [x2, y2] = layerPositions[layer2];
      const activity1 = layers[layer1]?.activity || 0;
      const activity2 = layers[layer2]?.activity || 0;
      const flowStrength = Math.abs(activity1 - activity2);
      
      if (flowStrength > 0.1) {
        ctx.strokeStyle = `rgba(0, 255, 255, ${flowStrength})`;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        
        // Draw flow particles
        const numParticles = Math.floor(flowStrength * 3) + 1;
        for (let i = 0; i < numParticles; i++) {
          const t = (i / numParticles + Date.now() / 1000) % 1.0;
          const px = x1 + t * (x2 - x1);
          const py = y1 + t * (y2 - y1);
          
          ctx.fillStyle = 'rgba(0, 255, 255, 0.8)';
          ctx.beginPath();
          ctx.arc(px, py, 4, 0, 2 * Math.PI);
          ctx.fill();
        }
      }
    });
    
    // Draw layer regions
    Object.entries(layerPositions).forEach(([layerName, [cx, cy]]) => {
      const state = layers[layerName] || { activity: 0, conflict: 0, pressure: 0 };
      
      // Pulsing effect
      const pulse = 1.0 + 0.2 * state.activity * Math.sin(Date.now() / 300);
      const currentRadius = radius * pulse;
      
      // Heat map color
      const activity = state.activity;
      let color;
      if (activity < 0.33) {
        const t = activity * 3;
        color = `rgba(${Math.floor(100 + t * 50)}, ${Math.floor(100 + t * 100)}, 255, 0.6)`;
      } else if (activity < 0.66) {
        const t = (activity - 0.33) * 3;
        color = `rgba(255, ${Math.floor(200 - t * 100)}, ${Math.floor(100 - t * 100)}, 0.6)`;
      } else {
        const t = (activity - 0.66) * 3;
        color = `rgba(255, ${Math.floor(100 - t * 50)}, 0, 0.6)`;
      }
      
      // Main region
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(cx, cy, currentRadius, 0, 2 * Math.PI);
      ctx.fill();
      
      // Conflict visualization
      if (state.conflict > 0.3) {
        const numPoints = 8;
        for (let i = 0; i < numPoints; i++) {
          const angle = (i / numPoints) * 2 * Math.PI + Date.now() / 500;
          const offset = state.conflict * 15;
          const px = cx + offset * Math.cos(angle);
          const py = cy + offset * Math.sin(angle);
          
          ctx.fillStyle = 'rgba(255, 255, 0, 0.7)';
          ctx.beginPath();
          ctx.arc(px, py, 3, 0, 2 * Math.PI);
          ctx.fill();
        }
      }
      
      // Pressure ring
      if (state.pressure > 0.2) {
        ctx.strokeStyle = `rgba(255, 0, 0, ${state.pressure * 0.8})`;
        ctx.lineWidth = state.pressure * 6;
        ctx.beginPath();
        ctx.arc(cx, cy, currentRadius * 1.1, 0, 2 * Math.PI);
        ctx.stroke();
      }
      
      // Layer label
      ctx.fillStyle = 'white';
      ctx.font = '12px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(layerName.toUpperCase(), cx, cy + currentRadius + 20);
    });
    
  }, [data]);
  
  return (
    <div className="relative">
      <canvas 
        ref={canvasRef} 
        width={800} 
        height={600}
        className="w-full h-auto bg-black rounded-lg border border-gray-800"
      />
      
      {/* Legend */}
      <div className="mt-4 grid grid-cols-4 gap-2 text-xs text-gray-400">
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 rounded-full bg-blue-500" />
          <span>Low Activity</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 rounded-full bg-yellow-500" />
          <span>Medium Activity</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 rounded-full bg-red-500" />
          <span>High Activity</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 rounded-full bg-yellow-300 animate-pulse" />
          <span>Conflict</span>
        </div>
      </div>
    </div>
  );
}

export default CognitivePhysiologyPlane;
