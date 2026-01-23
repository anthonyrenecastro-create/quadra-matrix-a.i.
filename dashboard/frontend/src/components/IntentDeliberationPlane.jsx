import React, { useEffect, useRef } from 'react';
import { useStore } from '../store/dashboardStore';

function IntentDeliberationPlane({ data }) {
  const canvasRef = useRef(null);
  const { sendInteraction } = useStore();
  
  useEffect(() => {
    if (!data || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.fillStyle = '#0f0f1a';
    ctx.fillRect(0, 0, width, height);
    
    const centerX = width / 2;
    const centerY = height / 2;
    const scale = Math.min(width, height) / 3;
    
    // Draw core objective
    ctx.fillStyle = 'rgba(255, 136, 0, 0.3)';
    ctx.beginPath();
    ctx.arc(centerX, centerY, 40, 0, 2 * Math.PI);
    ctx.fill();
    
    ctx.fillStyle = 'white';
    ctx.font = 'bold 14px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('CORE', centerX, centerY - 8);
    ctx.fillText('OBJECTIVE', centerX, centerY + 8);
    
    // Draw goals
    const goals = data.goals || [];
    goals.forEach((goal) => {
      const [gx, gy] = goal.position || [0, 0];
      const x = centerX + gx * scale;
      const y = centerY + gy * scale;
      const size = 20 + goal.confidence * 30;
      
      // Goal color
      let color;
      if (goal.pinned) {
        color = 'rgba(0, 255, 0, 0.9)';
      } else if (goal.confidence > 0.7) {
        color = 'rgba(255, 170, 0, 0.8)';
      } else {
        color = 'rgba(102, 136, 255, 0.5)';
      }
      
      // Draw goal node
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, size, 0, 2 * Math.PI);
      ctx.fill();
      
      // Draw competing hypotheses orbiting
      const hypotheses = goal.hypotheses || [];
      hypotheses.forEach((_, i) => {
        const angle = (i / hypotheses.length) * 2 * Math.PI + Date.now() / 1000;
        const orbitRadius = size + 15;
        const hx = x + orbitRadius * Math.cos(angle);
        const hy = y + orbitRadius * Math.sin(angle);
        
        ctx.fillStyle = 'rgba(0, 255, 255, 0.6)';
        ctx.beginPath();
        ctx.arc(hx, hy, 4, 0, 2 * Math.PI);
        ctx.fill();
      });
      
      // Connection to core
      ctx.strokeStyle = color.replace('0.9', '0.3').replace('0.8', '0.3').replace('0.5', '0.3');
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(x, y);
      ctx.stroke();
      ctx.setLineDash([]);
      
      // Goal name
      ctx.fillStyle = 'white';
      ctx.font = '11px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(goal.name, x, y + size + 15);
      
      // Confidence
      ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
      ctx.font = '9px monospace';
      ctx.fillText(`${(goal.confidence * 100).toFixed(0)}%`, x, y + size + 28);
    });
    
    // Draw confidence mass indicator
    const totalConf = goals.reduce((sum, g) => sum + g.confidence, 0);
    const avgConf = goals.length > 0 ? totalConf / goals.length : 0;
    
    ctx.fillStyle = 'rgba(255, 255, 0, 0.3)';
    ctx.fillRect(width - 30, 20, 20, height - 40);
    
    const confHeight = (height - 40) * avgConf;
    ctx.fillStyle = 'rgba(255, 255, 0, 0.7)';
    ctx.fillRect(width - 30, height - 20 - confHeight, 20, confHeight);
    
    ctx.fillStyle = 'white';
    ctx.font = '10px monospace';
    ctx.save();
    ctx.translate(width - 10, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('CONFIDENCE', 0, 0);
    ctx.restore();
    
  }, [data]);
  
  const handleCanvasClick = (event) => {
    if (!data?.goals) return;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;
    
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const scale = Math.min(canvas.width, canvas.height) / 3;
    
    // Check if clicked on any goal
    data.goals.forEach((goal) => {
      const [gx, gy] = goal.position || [0, 0];
      const goalX = centerX + gx * scale;
      const goalY = centerY + gy * scale;
      const size = 20 + goal.confidence * 30;
      
      const distance = Math.sqrt((x - goalX) ** 2 + (y - goalY) ** 2);
      if (distance < size) {
        // Pin this goal
        sendInteraction({
          command_type: 'pin_goal',
          target: goal.name
        });
      }
    });
  };
  
  return (
    <div className="relative">
      <canvas 
        ref={canvasRef} 
        width={800} 
        height={600}
        className="w-full h-auto bg-black rounded-lg border border-gray-800 cursor-pointer"
        onClick={handleCanvasClick}
      />
      
      {/* Instructions */}
      <div className="mt-4 p-3 bg-neural-dark rounded border border-gray-700">
        <p className="text-sm text-gray-400">
          <span className="text-neural-orange font-bold">Interactive:</span> Click on a goal to pin it (mark as high priority)
        </p>
        <div className="mt-2 grid grid-cols-3 gap-2 text-xs text-gray-400">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-green-500" />
            <span>Pinned Goal</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-yellow-600" />
            <span>High Confidence</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-blue-500" />
            <span>Emerging</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default IntentDeliberationPlane;
