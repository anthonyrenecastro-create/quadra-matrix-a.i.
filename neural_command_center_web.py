#!/usr/bin/env python3
"""
Neural Command Center - Web-Based Version
Three-Plane Architecture that works in browser/codespace environments

Accessible via browser at localhost:8050
"""

import torch
import numpy as np
import time
import math
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.figure import Figure
import matplotlib.animation as animation
from io import BytesIO
import base64

from demo_cognition import CognitionDemo

try:
    from flask import Flask, render_template_string, Response
    import threading
except ImportError:
    logger.error("Flask required. Install with: pip install flask")
    exit(1)


@dataclass
class GoalNode:
    """A goal in the deliberation space"""
    name: str
    position: Tuple[float, float]
    confidence: float = 0.5
    velocity: Tuple[float, float] = (0.0, 0.0)
    pinned: bool = False
    active: bool = True
    competing_hypotheses: List[str] = None
    
    def __post_init__(self):
        if self.competing_hypotheses is None:
            self.competing_hypotheses = []


@dataclass
class ActionVector:
    """A proposed action in the world interface"""
    name: str
    magnitude: float
    uncertainty: float
    outcome_estimate: str
    approved: bool = False
    timestamp: float = 0.0


class NeuralCommandCenterWeb:
    """Web-based Neural Command Center"""
    
    def __init__(self):
        self.demo = CognitionDemo(field_size=100, device='cpu')
        self.running = False
        self.pulse_phase = 0.0
        self.time_step = 0
        
        # Plane states
        self.layer_states = {
            'perception': {'activity': 0.0, 'conflict': 0.0, 'pressure': 0.0},
            'integration': {'activity': 0.0, 'conflict': 0.0, 'pressure': 0.0},
            'reasoning': {'activity': 0.0, 'conflict': 0.0, 'pressure': 0.0},
            'action': {'activity': 0.0, 'conflict': 0.0, 'pressure': 0.0}
        }
        
        self.goals = []
        self._spawn_initial_goals()
        
        self.sensors = {'visual': 0.0, 'temporal': 0.0, 'pattern': 0.0}
        self.proposed_actions = []
        self.action_history = deque(maxlen=20)
        
        # Create figure
        self.fig = Figure(figsize=(14, 12), facecolor='#000000')
        self.fig.subplots_adjust(hspace=0.3, left=0.05, right=0.95, top=0.95, bottom=0.05)
        
        self.ax_physiology = self.fig.add_subplot(3, 1, 1)
        self.ax_deliberation = self.fig.add_subplot(3, 1, 2)
        self.ax_world = self.fig.add_subplot(3, 1, 3)
    
    def _spawn_initial_goals(self):
        """Create initial goal nodes"""
        initial_goals = [
            ("Explore", 0.6, (0.5, 0.7)),
            ("Consolidate", 0.7, (-0.5, 0.6)),
            ("Predict", 0.5, (0.0, -0.7)),
            ("Adapt", 0.4, (-0.6, -0.3))
        ]
        
        for name, conf, pos in initial_goals:
            self.goals.append(GoalNode(
                name=name,
                position=pos,
                confidence=conf,
                velocity=(np.random.randn() * 0.01, np.random.randn() * 0.01),
                competing_hypotheses=[f"H{i}" for i in range(np.random.randint(1, 4))]
            ))
    
    def update_state(self):
        """Update the cognitive model state"""
        input_stimulus = torch.randn(100, device='cpu') * 0.1
        
        # Neural processing
        spikes1 = self.demo.oscillator.nn1(input_stimulus)
        spikes2 = self.demo.oscillator.nn2(input_stimulus)
        spikes3 = self.demo.oscillator.nn3(input_stimulus)
        
        # Field evolution
        combined_spikes = (spikes1 + spikes2 + spikes3) / 3
        self.demo.oscillator.field = 0.95 * self.demo.oscillator.field + 0.05 * combined_spikes
        
        self.pulse_phase = (self.pulse_phase + 0.1) % (2 * np.pi)
        self.time_step += 1
        
        self._compute_layer_states()
        self._update_goals()
        self._update_sensors()
        self._generate_actions()
    
    def _compute_layer_states(self):
        """Compute layer states from oscillator"""
        field = self.demo.oscillator.field
        quarter = field.shape[0] // 4
        
        layers_data = {
            'perception': field[:quarter],
            'integration': field[quarter:2*quarter],
            'reasoning': field[2*quarter:3*quarter],
            'action': field[3*quarter:]
        }
        
        for layer_name, layer_field in layers_data.items():
            activity = float(layer_field.abs().mean().item())
            conflict = float(layer_field.var().item())
            if len(layer_field) > 1:
                gradient = torch.diff(layer_field)
                pressure = float(gradient.abs().mean().item())
            else:
                pressure = 0.0
            
            self.layer_states[layer_name] = {
                'activity': np.clip(activity, 0, 1),
                'conflict': np.clip(conflict, 0, 1),
                'pressure': np.clip(pressure, 0, 1)
            }
    
    def _update_goals(self):
        """Update goal dynamics"""
        dt = 0.05
        
        for goal in self.goals:
            if goal.pinned or not goal.active:
                continue
            
            dx, dy = goal.position
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0.1:
                gravity = 0.3 / (dist ** 2)
                vx, vy = goal.velocity
                vx -= gravity * dx / dist * dt
                vy -= gravity * dy / dist * dt
                vx += 0.1 * (-dy / dist) * dt
                vy += 0.1 * (dx / dist) * dt
                vx *= 0.98
                vy *= 0.98
                goal.velocity = (vx, vy)
                goal.position = (dx + vx * dt, dy + vy * dt)
            
            goal.confidence += (np.random.rand() - 0.5) * 0.02
            goal.confidence = np.clip(goal.confidence, 0.1, 1.0)
        
        if len(self.goals) < 6 and np.random.rand() < 0.05:
            angle = np.random.rand() * 2 * np.pi
            radius = 0.6 + np.random.rand() * 0.3
            pos = (radius * np.cos(angle), radius * np.sin(angle))
            name = np.random.choice(["Discover", "Optimize", "Validate", "Transform"])
            self.goals.append(GoalNode(
                name=name, position=pos,
                confidence=0.3 + np.random.rand() * 0.3,
                velocity=(np.random.randn() * 0.02, np.random.randn() * 0.02),
                competing_hypotheses=[f"H{i}" for i in range(np.random.randint(1, 3))]
            ))
        
        self.goals = [g for g in self.goals if g.confidence > 0.15 or g.pinned]
    
    def _update_sensors(self):
        """Update sensor values"""
        field = self.demo.oscillator.field
        self.sensors['visual'] = float(field[:len(field)//3].mean().item())
        self.sensors['temporal'] = float(field.var().item())
        self.sensors['pattern'] = float(np.clip(1.0 - field.std().item(), 0, 1))
    
    def _generate_actions(self):
        """Generate action proposals"""
        self.proposed_actions = [a for a in self.proposed_actions 
                                if self.time_step - a.timestamp < 30]
        
        if np.random.rand() < 0.15 and len(self.proposed_actions) < 5:
            action_types = [
                ("Adjust Rate", "Optimize dynamics"),
                ("Explore", "Discover patterns"),
                ("Consolidate", "Strengthen connections"),
                ("Query", "Seek information"),
                ("Refine", "Improve accuracy")
            ]
            name, outcome = action_types[np.random.randint(0, len(action_types))]
            self.proposed_actions.append(ActionVector(
                name=name,
                magnitude=float(self.sensors['pattern'] * np.random.rand()),
                uncertainty=1.0 - self.sensors['pattern'],
                outcome_estimate=outcome,
                timestamp=self.time_step
            ))
    
    def render_frame(self):
        """Render current state to image"""
        # Clear all axes
        self.ax_physiology.clear()
        self.ax_deliberation.clear()
        self.ax_world.clear()
        
        # Render Plane 1: Cognitive Physiology
        self._render_physiology()
        
        # Render Plane 2: Intent & Deliberation
        self._render_deliberation()
        
        # Render Plane 3: World Interface
        self._render_world()
        
        # Convert to image
        buf = BytesIO()
        self.fig.savefig(buf, format='png', facecolor='#000000', dpi=100)
        buf.seek(0)
        return buf
    
    def _render_physiology(self):
        """Render cognitive physiology plane"""
        ax = self.ax_physiology
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_facecolor('#0a0a0a')
        ax.axis('off')
        ax.set_title('COGNITIVE PHYSIOLOGY', fontsize=12, fontweight='bold', color='#00ff88')
        
        layer_positions = {
            'perception': (0.25, 0.75),
            'integration': (0.75, 0.75),
            'reasoning': (0.25, 0.25),
            'action': (0.75, 0.25)
        }
        
        radius = 0.15
        
        # Draw connections
        connections = [
            ('perception', 'integration'),
            ('integration', 'reasoning'),
            ('reasoning', 'action'),
            ('perception', 'reasoning'),
            ('integration', 'action')
        ]
        
        for layer1, layer2 in connections:
            p1 = layer_positions[layer1]
            p2 = layer_positions[layer2]
            activity1 = self.layer_states[layer1]['activity']
            activity2 = self.layer_states[layer2]['activity']
            flow_strength = abs(activity1 - activity2)
            
            if flow_strength > 0.1:
                num_particles = int(flow_strength * 3) + 1
                for i in range(num_particles):
                    t = (i / num_particles + self.pulse_phase / (2*np.pi)) % 1.0
                    px = p1[0] + t * (p2[0] - p1[0])
                    py = p1[1] + t * (p2[1] - p1[1])
                    ax.plot(px, py, 'o', color='cyan', markersize=4, alpha=0.6)
        
        # Draw layers
        for layer_name, (cx, cy) in layer_positions.items():
            state = self.layer_states[layer_name]
            pulse_factor = 1.0 + 0.2 * state['activity'] * np.sin(self.pulse_phase)
            current_radius = radius * pulse_factor
            
            activity = state['activity']
            if activity < 0.33:
                color = plt.cm.Blues(0.3 + activity * 2)
            elif activity < 0.66:
                color = plt.cm.YlOrRd(activity)
            else:
                color = plt.cm.hot(activity)
            
            circle = Circle((cx, cy), current_radius, color=color, alpha=0.6, zorder=2)
            ax.add_patch(circle)
            
            if state['conflict'] > 0.3:
                for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                    offset = state['conflict'] * 0.05
                    px = cx + offset * np.cos(angle + self.pulse_phase)
                    py = cy + offset * np.sin(angle + self.pulse_phase)
                    ax.plot(px, py, 'o', color='yellow', markersize=3, alpha=0.7)
            
            if state['pressure'] > 0.2:
                ring = Circle((cx, cy), current_radius * 1.1,
                            fill=False, edgecolor='red',
                            linewidth=state['pressure'] * 4, alpha=0.5)
                ax.add_patch(ring)
    
    def _render_deliberation(self):
        """Render intent & deliberation plane"""
        ax = self.ax_deliberation
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.set_facecolor('#0f0f1a')
        ax.axis('off')
        ax.set_title('INTENT & DELIBERATION', fontsize=12, fontweight='bold', color='#ff8800')
        
        # Core objective
        ax.add_patch(Circle((0, 0), 0.15, color='orange', alpha=0.3))
        ax.text(0, 0, 'Core\nObjective', ha='center', va='center',
               fontsize=9, fontweight='bold', color='white')
        
        # Goals
        for goal in self.goals:
            x, y = goal.position
            size = 0.08 + goal.confidence * 0.12
            
            if goal.pinned:
                color, alpha = '#00ff00', 0.9
            elif goal.confidence > 0.7:
                color, alpha = '#ffaa00', 0.8
            else:
                color, alpha = '#6688ff', 0.5
            
            circle = Circle((x, y), size, color=color, alpha=alpha, zorder=3)
            ax.add_patch(circle)
            
            # Hypotheses
            for j in range(len(goal.competing_hypotheses)):
                angle = (j / len(goal.competing_hypotheses)) * 2 * np.pi + time.time()
                hx = x + (size + 0.05) * np.cos(angle)
                hy = y + (size + 0.05) * np.sin(angle)
                ax.plot(hx, hy, 'o', color='cyan', markersize=3, alpha=0.6)
            
            ax.plot([0, x], [0, y], '--', color=color, alpha=0.3, linewidth=1)
            
            # Label
            ax.text(x, y - size - 0.08, goal.name, ha='center', 
                   fontsize=7, color='white', alpha=0.8)
    
    def _render_world(self):
        """Render world interface plane"""
        ax = self.ax_world
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_facecolor('#1a0f0f')
        ax.axis('off')
        ax.set_title('WORLD INTERFACE', fontsize=12, fontweight='bold', color='#ff0088')
        
        # Sensors
        sensor_positions = {'visual': 0.8, 'temporal': 0.5, 'pattern': 0.2}
        for sensor_name, y_pos in sensor_positions.items():
            ax.text(0.05, y_pos, sensor_name.upper(), fontsize=8, color='white', va='center')
            value = self.sensors[sensor_name]
            bar_length = value * 0.25
            rect = Rectangle((0.18, y_pos - 0.02), bar_length, 0.04,
                           color='cyan', alpha=0.7)
            ax.add_patch(rect)
            ax.text(0.45, y_pos, f"{value:.2f}", fontsize=7, color='cyan', va='center')
        
        # Actions
        ax.text(0.55, 0.95, "PROPOSED ACTIONS", fontsize=9, fontweight='bold', color='#ff8800')
        
        start_y = 0.85
        for i, action in enumerate(self.proposed_actions[:5]):
            y = start_y - i * 0.15
            arrow_length = action.magnitude * 0.15
            
            arrow = FancyArrowPatch((0.55, y), (0.55 + arrow_length, y),
                                   arrowstyle='->', mutation_scale=15,
                                   color='yellow' if not action.approved else 'green',
                                   alpha=0.8, linewidth=2)
            ax.add_patch(arrow)
            
            if action.uncertainty > 0.3:
                cloud = Circle((0.55 + arrow_length/2, y), action.uncertainty * 0.08,
                             color='red', alpha=0.3)
                ax.add_patch(cloud)
            
            age = self.time_step - action.timestamp
            alpha = 1.0 - (age / 30)
            ax.text(0.56 + arrow_length, y + 0.02, action.name,
                   fontsize=7, color='white', alpha=alpha, va='bottom')


# Flask app
app = Flask(__name__)
center = NeuralCommandCenterWeb()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>⚡ Neural Command Center</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            background-color: #000;
            font-family: 'Courier New', monospace;
            color: #00ff88;
        }
        .header {
            text-align: center;
            padding: 20px;
            background-color: #0a0a0a;
            border-bottom: 2px solid #00ff88;
        }
        h1 {
            margin: 0;
            font-size: 2em;
            color: #00ff88;
        }
        .subtitle {
            color: #888;
            font-size: 0.9em;
            margin-top: 5px;
        }
        .container {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        #frame {
            max-width: 100%;
            height: auto;
            border: 2px solid #00ff88;
            box-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
        }
        .controls {
            margin-top: 20px;
            padding: 15px;
            background-color: #1a1a1a;
            border: 1px solid #00ff88;
            border-radius: 5px;
        }
        button {
            background-color: #00aa00;
            color: white;
            border: none;
            padding: 10px 20px;
            margin: 5px;
            font-family: 'Courier New', monospace;
            font-size: 1em;
            font-weight: bold;
            cursor: pointer;
            border-radius: 3px;
        }
        button:hover {
            background-color: #00ff00;
        }
        button.stop {
            background-color: #aa0000;
        }
        button.stop:hover {
            background-color: #ff0000;
        }
        .status {
            margin-top: 10px;
            padding: 10px;
            text-align: center;
            font-weight: bold;
        }
        .online { color: #00ff00; }
        .offline { color: #ff0000; }
    </style>
</head>
<body>
    <div class="header">
        <h1>⚡ NEURAL COMMAND CENTER</h1>
        <div class="subtitle">Three-Plane Cognitive Architecture • Not a Chatbot • A Living System</div>
    </div>
    
    <div class="container">
        <img id="frame" src="{{ url_for('video_feed') }}" />
        
        <div class="controls">
            <button onclick="start()">▶ ACTIVATE SYSTEM</button>
            <button class="stop" onclick="stop()">⏸ PAUSE</button>
            <div class="status">
                Status: <span id="status" class="offline">● OFFLINE</span>
            </div>
        </div>
        
        <div style="margin-top: 20px; color: #888; text-align: center; max-width: 800px;">
            <p><strong>Plane 1 (Top):</strong> Cognitive Physiology - Pulses, flows, pressure, heat</p>
            <p><strong>Plane 2 (Middle):</strong> Intent & Deliberation - Goals forming, competing, dissolving</p>
            <p><strong>Plane 3 (Bottom):</strong> World Interface - Sensors, proposed actions, uncertainty</p>
            <p style="margin-top: 15px; color: #00ff88;"><em>"If frozen for 10 seconds, it still feels alive"</em></p>
        </div>
    </div>
    
    <script>
        let running = false;
        
        function start() {
            fetch('/start');
            document.getElementById('status').className = 'online';
            document.getElementById('status').textContent = '● ONLINE';
            running = true;
        }
        
        function stop() {
            fetch('/stop');
            document.getElementById('status').className = 'offline';
            document.getElementById('status').textContent = '● PAUSED';
            running = false;
        }
        
        // Auto-refresh frame
        setInterval(() => {
            if (running) {
                document.getElementById('frame').src = '{{ url_for("video_feed") }}?' + new Date().getTime();
            }
        }, 200);  // ~5 FPS
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/start')
def start():
    center.running = True
    logger.info("🚀 Neural Command Center ACTIVATED")
    return "started"

@app.route('/stop')
def stop():
    center.running = False
    logger.info("⏸ Neural Command Center PAUSED")
    return "stopped"

@app.route('/video_feed')
def video_feed():
    if center.running:
        center.update_state()
    
    buf = center.render_frame()
    return Response(buf.getvalue(), mimetype='image/png')


def main():
    """Launch the web-based Neural Command Center"""
    logger.info("✨ Neural Command Center - Web Version")
    logger.info("   Three planes: Physiology | Deliberation | World Interface")
    logger.info("   Access via browser at: http://localhost:8050")
    logger.info("")
    
    app.run(host='0.0.0.0', port=8050, debug=False, threaded=True)


if __name__ == "__main__":
    main()
