#!/usr/bin/env python3
"""
Neural Command Center - Three-Plane Architecture
A living cognitive interface without chatboxes.

Three Always-Visible Planes:
1. Cognitive Physiology Plane: The heart monitor of intelligence
2. Intent & Deliberation Plane: Where goals form, compete, and dissolve
3. World Interface Plane: External coupling and action proposals
"""

import tkinter as tk
from tkinter import ttk
import torch
import numpy as np
import threading
import time
import math
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, Wedge
    import matplotlib.patches as mpatches
except ImportError:
    logger.error("matplotlib required. Install with: pip install matplotlib")
    exit(1)

from demo_cognition import CognitionDemo


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


class CognitivePhysiologyPlane:
    """
    Plane 1: The heart monitor of intelligence
    Shows activity as pulses, flows, pressure, heat
    """
    
    def __init__(self, ax, demo):
        self.ax = ax
        self.demo = demo
        self.layer_states = {
            'perception': {'activity': 0.0, 'conflict': 0.0, 'pressure': 0.0},
            'integration': {'activity': 0.0, 'conflict': 0.0, 'pressure': 0.0},
            'reasoning': {'activity': 0.0, 'conflict': 0.0, 'pressure': 0.0},
            'action': {'activity': 0.0, 'conflict': 0.0, 'pressure': 0.0}
        }
        
        self.activity_history = {layer: deque(maxlen=100) for layer in self.layer_states.keys()}
        self.pulse_phase = 0.0
        self.flow_particles = []
        
        # Initialize flow particles
        for i in range(20):
            self.flow_particles.append({
                'x': np.random.rand(),
                'y': np.random.rand(),
                'vx': (np.random.rand() - 0.5) * 0.02,
                'vy': (np.random.rand() - 0.5) * 0.02,
                'intensity': np.random.rand()
            })
        
        self.setup_axes()
    
    def setup_axes(self):
        """Setup the physiology visualization"""
        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#0a0a0a')
        self.ax.axis('off')
        self.ax.set_title('COGNITIVE PHYSIOLOGY', 
                         fontsize=10, fontweight='bold', 
                         color='#00ff88', pad=10)
    
    def update(self):
        """Update the physiology visualization"""
        self.ax.clear()
        self.setup_axes()
        
        # Update layer states from demo
        self._compute_layer_states()
        
        # Draw four interconnected regions (layers)
        layer_positions = {
            'perception': (0.25, 0.75),
            'integration': (0.75, 0.75),
            'reasoning': (0.25, 0.25),
            'action': (0.75, 0.25)
        }
        
        radius = 0.15
        
        # Draw connections (flows between layers)
        self._draw_flows(layer_positions, radius)
        
        # Draw layer regions
        for layer_name, (cx, cy) in layer_positions.items():
            state = self.layer_states[layer_name]
            self._draw_layer_region(cx, cy, radius, layer_name, state)
        
        # Update pulse phase
        self.pulse_phase = (self.pulse_phase + 0.1) % (2 * np.pi)
    
    def _compute_layer_states(self):
        """Compute current state of each cognitive layer"""
        # Simulate layer activity from oscillator
        if hasattr(self.demo.oscillator, 'field'):
            field = self.demo.oscillator.field
            
            # Partition field into layers
            quarter = field.shape[0] // 4
            
            layers_data = {
                'perception': field[:quarter],
                'integration': field[quarter:2*quarter],
                'reasoning': field[2*quarter:3*quarter],
                'action': field[3*quarter:]
            }
            
            for layer_name, layer_field in layers_data.items():
                # Activity: mean absolute activation
                activity = float(layer_field.abs().mean().item())
                
                # Conflict: variance (how much disagreement)
                conflict = float(layer_field.var().item())
                
                # Pressure: gradient magnitude (rate of change)
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
                
                self.activity_history[layer_name].append(activity)
    
    def _draw_layer_region(self, cx, cy, radius, name, state):
        """Draw a single cognitive layer region"""
        # Pulsing effect based on activity
        pulse_factor = 1.0 + 0.2 * state['activity'] * np.sin(self.pulse_phase)
        current_radius = radius * pulse_factor
        
        # Heat map color (activity level)
        activity = state['activity']
        # Color gradient: blue (low) -> cyan -> yellow -> red (high)
        if activity < 0.33:
            color = plt.cm.Blues(0.3 + activity * 2)
        elif activity < 0.66:
            color = plt.cm.YlOrRd(activity)
        else:
            color = plt.cm.hot(activity)
        
        # Main region circle
        circle = Circle((cx, cy), current_radius, 
                       color=color, alpha=0.6, zorder=2)
        self.ax.add_patch(circle)
        
        # Conflict visualization (inner turbulence)
        if state['conflict'] > 0.3:
            conflict_points = 8
            angles = np.linspace(0, 2*np.pi, conflict_points, endpoint=False)
            for angle in angles:
                offset = state['conflict'] * 0.05
                px = cx + offset * np.cos(angle + self.pulse_phase)
                py = cy + offset * np.sin(angle + self.pulse_phase)
                self.ax.plot(px, py, 'o', color='yellow', 
                           markersize=3, alpha=0.7, zorder=3)
        
        # Pressure indicator (outer ring)
        if state['pressure'] > 0.2:
            ring = Circle((cx, cy), current_radius * 1.1,
                         fill=False, edgecolor='red', 
                         linewidth=state['pressure'] * 4,
                         alpha=0.5, zorder=1)
            self.ax.add_patch(ring)
        
        # Memory activation spikes (random bursts)
        if np.random.rand() < state['activity'] * 0.3:
            spike_angle = np.random.rand() * 2 * np.pi
            spike_len = 0.08 * state['activity']
            sx = cx + current_radius * np.cos(spike_angle)
            sy = cy + current_radius * np.sin(spike_angle)
            ex = sx + spike_len * np.cos(spike_angle)
            ey = sy + spike_len * np.sin(spike_angle)
            self.ax.plot([sx, ex], [sy, ey], 'w-', linewidth=2, alpha=0.8, zorder=4)
    
    def _draw_flows(self, positions, radius):
        """Draw energy flows between layers"""
        connections = [
            ('perception', 'integration'),
            ('integration', 'reasoning'),
            ('reasoning', 'action'),
            ('perception', 'reasoning'),
            ('integration', 'action')
        ]
        
        for layer1, layer2 in connections:
            p1 = positions[layer1]
            p2 = positions[layer2]
            
            # Flow strength based on activity difference
            activity1 = self.layer_states[layer1]['activity']
            activity2 = self.layer_states[layer2]['activity']
            flow_strength = abs(activity1 - activity2)
            
            if flow_strength > 0.1:
                # Animated flow particles
                num_particles = int(flow_strength * 5) + 1
                for i in range(num_particles):
                    t = (i / num_particles + self.pulse_phase / (2*np.pi)) % 1.0
                    px = p1[0] + t * (p2[0] - p1[0])
                    py = p1[1] + t * (p2[1] - p1[1])
                    
                    self.ax.plot(px, py, 'o', color='cyan', 
                               markersize=4, alpha=0.6, zorder=1)


class IntentDeliberationPlane:
    """
    Plane 2: Where humans realize it's not a chatbot
    Goal nodes forming/dissolving, competing hypotheses, collaborative interaction
    """
    
    def __init__(self, ax, demo):
        self.ax = ax
        self.demo = demo
        self.goals = []
        self.selected_goal = None
        self.core_objective = "Optimize Learning"
        self.confidence_mass = 0.5
        self.instability_zones = []
        
        # Initialize some goals
        self._spawn_initial_goals()
    
    def setup_axes(self):
        """Setup the deliberation visualization"""
        self.ax.clear()
        self.ax.set_xlim(-1.2, 1.2)
        self.ax.set_ylim(-1.2, 1.2)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#0f0f1a')
        self.ax.axis('off')
        self.ax.set_title('INTENT & DELIBERATION', 
                         fontsize=10, fontweight='bold', 
                         color='#ff8800', pad=10)
    
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
    
    def update(self):
        """Update the deliberation visualization"""
        self.ax.clear()
        self.setup_axes()
        
        # Draw core objective (center)
        self.ax.add_patch(Circle((0, 0), 0.15, color='orange', alpha=0.3, zorder=1))
        self.ax.text(0, 0, self.core_objective, ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white', zorder=2)
        
        # Update and draw goal nodes
        self._update_goal_dynamics()
        self._draw_goals()
        
        # Draw confidence mass distribution
        self._draw_confidence_field()
        
        # Highlight instability zones
        self._draw_instability_zones()
    
    def _update_goal_dynamics(self):
        """Update goal positions and states (orbital motion, attraction/repulsion)"""
        dt = 0.05
        
        for goal in self.goals:
            if goal.pinned or not goal.active:
                continue
            
            # Gravity toward center (core objective)
            dx, dy = goal.position
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0.1:
                # Orbital mechanics
                gravity = 0.3 / (dist ** 2)
                vx, vy = goal.velocity
                vx -= gravity * dx / dist * dt
                vy -= gravity * dy / dist * dt
                
                # Orbital velocity (perpendicular to radius)
                vx += 0.1 * (-dy / dist) * dt
                vy += 0.1 * (dx / dist) * dt
                
                # Damping
                vx *= 0.98
                vy *= 0.98
                
                goal.velocity = (vx, vy)
                
                # Update position
                new_x = dx + vx * dt
                new_y = dy + vy * dt
                goal.position = (new_x, new_y)
            
            # Confidence fluctuation
            goal.confidence += (np.random.rand() - 0.5) * 0.02
            goal.confidence = np.clip(goal.confidence, 0.1, 1.0)
        
        # Goal lifecycle: spawn new or dissolve low confidence
        if len(self.goals) < 6 and np.random.rand() < 0.05:
            self._spawn_goal()
        
        self.goals = [g for g in self.goals if g.confidence > 0.15 or g.pinned]
    
    def _spawn_goal(self):
        """Spawn a new goal node"""
        angle = np.random.rand() * 2 * np.pi
        radius = 0.6 + np.random.rand() * 0.3
        pos = (radius * np.cos(angle), radius * np.sin(angle))
        
        goal_names = ["Discover", "Optimize", "Validate", "Transform", "Synthesize", "Analyze"]
        name = np.random.choice(goal_names)
        
        self.goals.append(GoalNode(
            name=name,
            position=pos,
            confidence=0.3 + np.random.rand() * 0.3,
            velocity=(np.random.randn() * 0.02, np.random.randn() * 0.02),
            competing_hypotheses=[f"H{i}" for i in range(np.random.randint(1, 4))]
        ))
    
    def _draw_goals(self):
        """Draw goal nodes with their hypotheses"""
        for i, goal in enumerate(self.goals):
            x, y = goal.position
            
            # Size based on confidence
            size = 0.08 + goal.confidence * 0.12
            
            # Color based on state
            if goal.pinned:
                color = '#00ff00'
                alpha = 0.9
            elif goal.confidence > 0.7:
                color = '#ffaa00'
                alpha = 0.8
            else:
                color = '#6688ff'
                alpha = 0.5
            
            # Draw node
            circle = Circle((x, y), size, color=color, alpha=alpha, zorder=3)
            self.ax.add_patch(circle)
            
            # Draw competing hypotheses orbiting
            num_hyp = len(goal.competing_hypotheses)
            for j in range(num_hyp):
                angle = (j / num_hyp) * 2 * np.pi + time.time()
                hx = x + (size + 0.05) * np.cos(angle)
                hy = y + (size + 0.05) * np.sin(angle)
                self.ax.plot(hx, hy, 'o', color='cyan', markersize=3, alpha=0.6, zorder=2)
            
            # Connection to core
            self.ax.plot([0, x], [0, y], '--', color=color, 
                        alpha=0.3, linewidth=1, zorder=1)
    
    def _draw_confidence_field(self):
        """Visualize confidence mass distribution"""
        # Create a heatmap of confidence in the space
        total_conf = sum(g.confidence for g in self.goals)
        if total_conf > 0:
            self.confidence_mass = total_conf / len(self.goals) if self.goals else 0.5
        
        # Draw confidence indicator
        conf_bar_x = 0.9
        conf_bar_y = np.linspace(-0.8, 0.8, 100)
        for i, y in enumerate(conf_bar_y):
            if i < len(conf_bar_y) * self.confidence_mass:
                self.ax.plot(conf_bar_x, y, 's', color='yellow', 
                           markersize=2, alpha=0.5)
    
    def _draw_instability_zones(self):
        """Highlight zones where decisions might be unstable"""
        # Find clusters of conflicting goals
        if len(self.goals) >= 2:
            for i, g1 in enumerate(self.goals):
                for g2 in self.goals[i+1:]:
                    x1, y1 = g1.position
                    x2, y2 = g2.position
                    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    
                    # If goals are close and competing
                    if dist < 0.3 and abs(g1.confidence - g2.confidence) < 0.2:
                        # Mark instability zone
                        mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
                        self.ax.add_patch(Circle((mid_x, mid_y), 0.1, 
                                                color='red', alpha=0.2, zorder=0))
    
    def pin_goal(self, goal_name: str):
        """Human interaction: pin a goal as important"""
        for goal in self.goals:
            if goal.name == goal_name:
                goal.pinned = True
                goal.confidence = 1.0
                logger.info(f"📌 Goal '{goal_name}' pinned by human")
    
    def inject_constraint(self, constraint: str):
        """Human interaction: inject a constraint"""
        logger.info(f"⚠️  Constraint injected: {constraint}")
        # This would affect goal weights and layer behavior
        for goal in self.goals:
            if constraint.lower() in goal.name.lower():
                goal.confidence *= 1.2
                goal.confidence = min(1.0, goal.confidence)


class WorldInterfacePlane:
    """
    Plane 3: External coupling and action proposals
    Sensors, actuators, proposed actions with uncertainty
    """
    
    def __init__(self, ax, demo):
        self.ax = ax
        self.demo = demo
        self.sensors = {
            'visual': 0.0,
            'temporal': 0.0,
            'pattern': 0.0
        }
        self.proposed_actions = []
        self.action_history = deque(maxlen=20)
        self.time_step = 0
    
    def setup_axes(self):
        """Setup the world interface visualization"""
        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#1a0f0f')
        self.ax.axis('off')
        self.ax.set_title('WORLD INTERFACE', 
                         fontsize=10, fontweight='bold', 
                         color='#ff0088', pad=10)
    
    def update(self):
        """Update the world interface visualization"""
        self.ax.clear()
        self.setup_axes()
        self.time_step += 1
        
        # Update sensors
        self._update_sensors()
        self._draw_sensors()
        
        # Generate and draw action proposals
        self._generate_action_proposals()
        self._draw_action_proposals()
        
        # Draw action history
        self._draw_action_history()
    
    def _update_sensors(self):
        """Update sensor readings from the cognitive system"""
        # Simulate sensor inputs from oscillator state
        if hasattr(self.demo.oscillator, 'field'):
            field = self.demo.oscillator.field
            
            # Visual sensor: mean activation in upper portion
            self.sensors['visual'] = float(field[:len(field)//3].mean().item())
            
            # Temporal sensor: gradient (change over time simulated by variance)
            self.sensors['temporal'] = float(field.var().item())
            
            # Pattern sensor: coherence
            self.sensors['pattern'] = float((1.0 - field.std().item()).clip(0, 1))
    
    def _draw_sensors(self):
        """Draw sensor inputs (left side)"""
        sensor_y_positions = {'visual': 0.8, 'temporal': 0.5, 'pattern': 0.2}
        
        for sensor_name, y_pos in sensor_y_positions.items():
            value = self.sensors[sensor_name]
            
            # Sensor label
            self.ax.text(0.05, y_pos, sensor_name.upper(), 
                        fontsize=8, color='white', va='center')
            
            # Sensor bar
            bar_length = value * 0.25
            rect = Rectangle((0.18, y_pos - 0.02), bar_length, 0.04,
                           color='cyan', alpha=0.7, zorder=2)
            self.ax.add_patch(rect)
            
            # Value text
            self.ax.text(0.45, y_pos, f"{value:.2f}", 
                        fontsize=7, color='cyan', va='center')
    
    def _generate_action_proposals(self):
        """Generate action proposals based on current state"""
        # Keep recent proposals, remove old ones
        self.proposed_actions = [a for a in self.proposed_actions 
                                if self.time_step - a.timestamp < 30]
        
        # Occasionally generate new proposals
        if np.random.rand() < 0.15 and len(self.proposed_actions) < 5:
            action_types = [
                ("Adjust Learning Rate", "Optimize training dynamics"),
                ("Explore New State", "Discover novel patterns"),
                ("Consolidate Memory", "Strengthen important connections"),
                ("Query External", "Seek additional information"),
                ("Refine Hypothesis", "Improve prediction accuracy")
            ]
            
            name, outcome = action_types[np.random.randint(0, len(action_types))]
            
            # Magnitude based on current system state
            magnitude = float(self.sensors['pattern'] * np.random.rand())
            
            # Uncertainty inversely related to confidence
            uncertainty = 1.0 - self.sensors['pattern']
            
            action = ActionVector(
                name=name,
                magnitude=magnitude,
                uncertainty=uncertainty,
                outcome_estimate=outcome,
                timestamp=self.time_step
            )
            
            self.proposed_actions.append(action)
    
    def _draw_action_proposals(self):
        """Draw proposed actions (center-right)"""
        start_y = 0.85
        spacing = 0.15
        
        self.ax.text(0.55, 0.95, "PROPOSED ACTIONS", 
                    fontsize=8, fontweight='bold', color='#ff8800')
        
        for i, action in enumerate(self.proposed_actions[:5]):
            y = start_y - i * spacing
            
            # Action vector arrow
            arrow_length = action.magnitude * 0.15
            arrow = FancyArrowPatch((0.55, y), (0.55 + arrow_length, y),
                                   arrowstyle='->', mutation_scale=15,
                                   color='yellow' if not action.approved else 'green',
                                   alpha=0.8, linewidth=2, zorder=3)
            self.ax.add_patch(arrow)
            
            # Uncertainty cloud
            if action.uncertainty > 0.3:
                cloud = Circle((0.55 + arrow_length/2, y), 
                             action.uncertainty * 0.08,
                             color='red', alpha=0.3, zorder=1)
                self.ax.add_patch(cloud)
            
            # Action name (on hover in real implementation)
            age = self.time_step - action.timestamp
            alpha = 1.0 - (age / 30)
            self.ax.text(0.56 + arrow_length, y + 0.02, 
                        action.name[:12], 
                        fontsize=6, color='white', alpha=alpha, va='bottom')
    
    def _draw_action_history(self):
        """Draw timeline of executed actions"""
        if len(self.action_history) > 0:
            self.ax.text(0.55, 0.15, "HISTORY", 
                        fontsize=7, color='#666666')
            
            for i, action in enumerate(list(self.action_history)[-5:]):
                y = 0.12 - i * 0.02
                self.ax.plot([0.55, 0.55 + action.magnitude * 0.1], [y, y],
                           '-', color='gray', alpha=0.5, linewidth=1)
    
    def approve_action(self, action_name: str):
        """Human approves an action"""
        for action in self.proposed_actions:
            if action.name == action_name:
                action.approved = True
                self.action_history.append(action)
                logger.info(f"✓ Action approved: {action_name}")
                break
    
    def sandbox_action(self, action_name: str):
        """Sandbox an action to test outcomes without executing"""
        logger.info(f"🧪 Sandboxing action: {action_name}")
        # Would run simulation/prediction


class InteractionPanel:
    """
    Non-conversational interaction modes:
    - Signal injection
    - Cognitive probes  
    - Ethical governors
    """
    
    def __init__(self, parent_frame, deliberation_plane, world_plane):
        self.frame = tk.LabelFrame(parent_frame, text="⚡ INTERACTION MODES", 
                                  font=('Courier', 9, 'bold'),
                                  bg='#1a1a1a', fg='#00ff88', 
                                  relief=tk.RAISED, borderwidth=2)
        self.frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        self.deliberation_plane = deliberation_plane
        self.world_plane = world_plane
        
        self._create_controls()
    
    def _create_controls(self):
        """Create interaction controls"""
        # Signal Injection
        injection_frame = tk.LabelFrame(self.frame, text="Signal Injection",
                                       bg='#2a2a2a', fg='#ffffff',
                                       font=('Courier', 8))
        injection_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(injection_frame, text="Sensory Stream:", 
                bg='#2a2a2a', fg='#aaaaaa', font=('Courier', 7)).pack(anchor=tk.W)
        
        sensor_var = tk.StringVar(value="pattern")
        for sensor in ['visual', 'temporal', 'pattern']:
            tk.Radiobutton(injection_frame, text=sensor.capitalize(),
                          variable=sensor_var, value=sensor,
                          bg='#2a2a2a', fg='#ffffff', 
                          selectcolor='#3a3a3a',
                          font=('Courier', 7)).pack(anchor=tk.W)
        
        tk.Button(injection_frame, text="Inject Data Pulse", 
                 bg='#0066cc', fg='white', font=('Courier', 7),
                 command=lambda: logger.info(f"💉 Injected {sensor_var.get()} pulse")).pack(pady=5)
        
        # Cognitive Probes
        probe_frame = tk.LabelFrame(self.frame, text="Cognitive Probes",
                                   bg='#2a2a2a', fg='#ffffff',
                                   font=('Courier', 8))
        probe_frame.pack(fill=tk.X, padx=5, pady=5)
        
        probe_buttons = [
            ("Show Uncertainty", "🔍 Uncertainty regions highlighted"),
            ("Find Conflicts", "⚠️  Showing conflicting assumptions"),
            ("Stability Test", "📊 Testing stability boundaries")
        ]
        
        for label, log_msg in probe_buttons:
            tk.Button(probe_frame, text=label,
                     bg='#663399', fg='white', font=('Courier', 7),
                     command=lambda msg=log_msg: logger.info(msg)).pack(fill=tk.X, pady=2)
        
        # Ethical Governors
        governor_frame = tk.LabelFrame(self.frame, text="Safety Governors",
                                      bg='#2a2a2a', fg='#ffffff',
                                      font=('Courier', 8))
        governor_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(governor_frame, text="Hard Boundaries:",
                bg='#2a2a2a', fg='#ff4444', font=('Courier', 7, 'bold')).pack()
        
        tk.Entry(governor_frame, bg='#3a3a3a', fg='white',
                font=('Courier', 7), width=20).pack(pady=2)
        
        tk.Button(governor_frame, text="Set Constraint",
                 bg='#cc0000', fg='white', font=('Courier', 7),
                 command=lambda: self.deliberation_plane.inject_constraint("safety")).pack(pady=2)
        
        # Goal Pinning
        pin_frame = tk.LabelFrame(self.frame, text="Goal Management",
                                 bg='#2a2a2a', fg='#ffffff',
                                 font=('Courier', 8))
        pin_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(pin_frame, text="📌 Pin Active Goal",
                 bg='#009900', fg='white', font=('Courier', 7),
                 command=self._pin_random_goal).pack(fill=tk.X, pady=2)
        
        tk.Button(pin_frame, text="✓ Approve Action",
                 bg='#006600', fg='white', font=('Courier', 7),
                 command=self._approve_random_action).pack(fill=tk.X, pady=2)
    
    def _pin_random_goal(self):
        """Pin a random active goal"""
        if self.deliberation_plane.goals:
            goal = np.random.choice(self.deliberation_plane.goals)
            self.deliberation_plane.pin_goal(goal.name)
    
    def _approve_random_action(self):
        """Approve a random proposed action"""
        if self.world_plane.proposed_actions:
            action = self.world_plane.proposed_actions[0]
            self.world_plane.approve_action(action.name)


class NeuralCommandCenter:
    """
    The Neural Command Center - Three Planes Architecture
    A living cognitive interface
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("⚡ Neural Command Center")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#000000')
        
        # Initialize cognitive demo
        self.demo = CognitionDemo(field_size=100, device='cpu')
        
        # Animation control
        self.running = False
        self.animation_id = None
        
        self._create_ui()
        
    def _create_ui(self):
        """Create the three-plane interface"""
        # Title Bar
        title_frame = tk.Frame(self.root, bg='#000000', height=60)
        title_frame.pack(fill=tk.X)
        
        title_label = tk.Label(
            title_frame,
            text="⚡ NEURAL COMMAND CENTER",
            font=('Courier', 24, 'bold'),
            fg='#00ff88',
            bg='#000000'
        )
        title_label.pack(pady=10)
        
        subtitle_label = tk.Label(
            title_frame,
            text="Cognitive Architecture • Not a Chatbot • A Living System",
            font=('Courier', 10),
            fg='#888888',
            bg='#000000'
        )
        subtitle_label.pack()
        
        # Main content area (three planes + interaction panel)
        content_frame = tk.Frame(self.root, bg='#000000')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left: Three planes (stacked but all visible)
        planes_frame = tk.Frame(content_frame, bg='#000000')
        planes_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure with 3 subplots
        self.fig = Figure(figsize=(12, 10), facecolor='#000000')
        self.fig.subplots_adjust(hspace=0.3, left=0.05, right=0.95, top=0.95, bottom=0.05)
        
        # Three planes
        self.ax_physiology = self.fig.add_subplot(3, 1, 1)
        self.ax_deliberation = self.fig.add_subplot(3, 1, 2)
        self.ax_world = self.fig.add_subplot(3, 1, 3)
        
        # Initialize planes
        self.physiology_plane = CognitivePhysiologyPlane(self.ax_physiology, self.demo)
        self.deliberation_plane = IntentDeliberationPlane(self.ax_deliberation, self.demo)
        self.world_plane = WorldInterfacePlane(self.ax_world, self.demo)
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=planes_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Right: Interaction panel
        self.interaction_panel = InteractionPanel(
            content_frame, 
            self.deliberation_plane, 
            self.world_plane
        )
        
        # Control bar at bottom
        control_frame = tk.Frame(self.root, bg='#1a1a1a', height=50)
        control_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.start_btn = tk.Button(
            control_frame,
            text="▶ ACTIVATE SYSTEM",
            command=self.start,
            font=('Courier', 12, 'bold'),
            bg='#00aa00',
            fg='white',
            padx=20,
            pady=10
        )
        self.start_btn.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.stop_btn = tk.Button(
            control_frame,
            text="⏸ PAUSE",
            command=self.stop,
            font=('Courier', 12, 'bold'),
            bg='#aa0000',
            fg='white',
            padx=20,
            pady=10,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5, pady=10)
        
        # Status indicator
        self.status_label = tk.Label(
            control_frame,
            text="● OFFLINE",
            font=('Courier', 11, 'bold'),
            fg='#ff0000',
            bg='#1a1a1a'
        )
        self.status_label.pack(side=tk.RIGHT, padx=20)
        
    def start(self):
        """Activate the neural command center"""
        if self.running:
            return
            
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="● ONLINE", fg='#00ff00')
        
        logger.info("🚀 Neural Command Center ACTIVATED")
        self._animate()
    
    def stop(self):
        """Pause the system"""
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="● PAUSED", fg='#ffaa00')
        
        if self.animation_id:
            self.root.after_cancel(self.animation_id)
        
        logger.info("⏸ Neural Command Center PAUSED")
    
    def _animate(self):
        """Continuous animation loop - the 'living system' heartbeat"""
        if not self.running:
            return
        
        # Update demo state
        self._update_demo_state()
        
        # Update all three planes
        self.physiology_plane.update()
        self.deliberation_plane.update()
        self.world_plane.update()
        
        # Redraw
        self.canvas.draw()
        
        # Schedule next frame (60 FPS ≈ 16ms)
        self.animation_id = self.root.after(100, self._animate)
    
    def _update_demo_state(self):
        """Update the underlying cognitive model state"""
        # Run oscillator forward
        input_stimulus = torch.randn(100, device='cpu') * 0.1
        
        # Neural processing
        spikes1 = self.demo.oscillator.nn1(input_stimulus)
        spikes2 = self.demo.oscillator.nn2(input_stimulus)
        spikes3 = self.demo.oscillator.nn3(input_stimulus)
        
        # Field evolution
        combined_spikes = (spikes1 + spikes2 + spikes3) / 3
        self.demo.oscillator.field = 0.95 * self.demo.oscillator.field + 0.05 * combined_spikes
        
        # Memory updates
        if self.demo.memory.consolidated_state is None:
            self.demo.memory.consolidated_state = self.demo.oscillator.field.clone()
        else:
            self.demo.memory.consolidated_state = 0.9 * self.demo.memory.consolidated_state + 0.1 * self.demo.oscillator.field


def main():
    """Launch the Neural Command Center"""
    root = tk.Tk()
    center = NeuralCommandCenter(root)
    
    # Center window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'+{x}+{y}')
    
    logger.info("✨ Neural Command Center initialized")
    logger.info("   Three planes: Physiology | Deliberation | World Interface")
    logger.info("   Interaction modes: Signal Injection | Cognitive Probes | Safety Governors")
    
    root.mainloop()


if __name__ == "__main__":
    main()
