"""
CognitionSim Neural Command Center - FastAPI Backend
Production-grade backend with REST + WebSocket support for live cognition streaming

Features:
- Real-time cognition state via WebSockets
- Training metrics streaming
- Governance & policy management
- Edge deployment status
- Multi-user role-based access
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import asyncio
import json
import torch
import numpy as np
from datetime import datetime
from collections import deque
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from demo_cognition import CognitionDemo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="CognitionSim Neural Command Center",
    description="Advanced cognitive architecture dashboard with live streaming",
    version="2.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Data Models ====================

class CognitionState(BaseModel):
    """Current cognitive state snapshot"""
    timestamp: float
    layers: Dict[str, Dict[str, float]]  # layer_name -> {activity, conflict, pressure}
    goals: List[Dict[str, Any]]
    sensors: Dict[str, float]
    proposed_actions: List[Dict[str, Any]]
    field_coherence: float
    memory_magnitude: float


class TrainingMetrics(BaseModel):
    """Training progress metrics"""
    epoch: int
    step: int
    loss: float
    perplexity: Optional[float]
    gradient_norm: Optional[float]
    learning_rate: float
    timestamp: float


class GovernancePolicy(BaseModel):
    """Governance policy definition"""
    policy_id: str
    name: str
    type: str  # "hard_boundary" | "soft_penalty" | "contextual"
    description: str
    enabled: bool
    parameters: Dict[str, Any]


class EdgeDeployment(BaseModel):
    """Edge deployment status"""
    device_id: str
    status: str  # "online" | "offline" | "degraded"
    model_version: str
    inference_count: int
    avg_latency_ms: float
    memory_usage_mb: float
    last_heartbeat: float


class InteractionCommand(BaseModel):
    """User interaction command"""
    command_type: str  # "pin_goal" | "inject_constraint" | "approve_action" | "probe"
    target: Optional[str]
    parameters: Optional[Dict[str, Any]]


# ==================== Global State ====================

class DashboardState:
    """Central dashboard state manager"""
    
    def __init__(self):
        self.demo = CognitionDemo(field_size=100, device='cpu')
        self.running = False
        self.websocket_clients: List[WebSocket] = []
        
        # Cognition state
        self.pulse_phase = 0.0
        self.time_step = 0
        self.layer_states = {
            'perception': {'activity': 0.0, 'conflict': 0.0, 'pressure': 0.0},
            'integration': {'activity': 0.0, 'conflict': 0.0, 'pressure': 0.0},
            'reasoning': {'activity': 0.0, 'conflict': 0.0, 'pressure': 0.0},
            'action': {'activity': 0.0, 'conflict': 0.0, 'pressure': 0.0}
        }
        self.goals = []
        self.sensors = {'visual': 0.0, 'temporal': 0.0, 'pattern': 0.0}
        self.proposed_actions = []
        
        # Training state
        self.training_metrics = deque(maxlen=1000)
        self.is_training = False
        
        # Governance
        self.policies: Dict[str, GovernancePolicy] = {}
        self._load_default_policies()
        
        # Edge deployments
        self.edge_devices: Dict[str, EdgeDeployment] = {}
        
        self._initialize_goals()
    
    def _load_default_policies(self):
        """Load default governance policies"""
        default_policies = [
            GovernancePolicy(
                policy_id="safety_01",
                name="Memory Bounds",
                type="hard_boundary",
                description="Prevent memory overflow",
                enabled=True,
                parameters={"max_memory_mb": 4096}
            ),
            GovernancePolicy(
                policy_id="ethical_01",
                name="Fairness Constraint",
                type="soft_penalty",
                description="Penalize biased predictions",
                enabled=True,
                parameters={"threshold": 0.15}
            )
        ]
        for policy in default_policies:
            self.policies[policy.policy_id] = policy
    
    def _initialize_goals(self):
        """Initialize goal nodes"""
        initial_goals = [
            {"name": "Explore", "confidence": 0.6, "position": [0.5, 0.7], 
             "pinned": False, "hypotheses": ["H1", "H2"]},
            {"name": "Consolidate", "confidence": 0.7, "position": [-0.5, 0.6],
             "pinned": False, "hypotheses": ["H1", "H2", "H3"]},
            {"name": "Predict", "confidence": 0.5, "position": [0.0, -0.7],
             "pinned": False, "hypotheses": ["H1"]},
            {"name": "Adapt", "confidence": 0.4, "position": [-0.6, -0.3],
             "pinned": False, "hypotheses": ["H1", "H2"]}
        ]
        self.goals = initial_goals
    
    def update_cognition(self):
        """Update cognitive state - called in background loop"""
        # Neural processing
        input_stimulus = torch.randn(100, device='cpu') * 0.1
        spikes1 = self.demo.oscillator.nn1(input_stimulus)
        spikes2 = self.demo.oscillator.nn2(input_stimulus)
        spikes3 = self.demo.oscillator.nn3(input_stimulus)
        
        # Field evolution
        combined_spikes = (spikes1 + spikes2 + spikes3) / 3
        self.demo.oscillator.field = 0.95 * self.demo.oscillator.field + 0.05 * combined_spikes
        
        self.pulse_phase = (self.pulse_phase + 0.1) % (2 * np.pi)
        self.time_step += 1
        
        # Update layers
        self._compute_layer_states()
        
        # Update goals (orbital dynamics)
        self._update_goals()
        
        # Update sensors
        self._update_sensors()
        
        # Generate actions
        self._generate_actions()
    
    def _compute_layer_states(self):
        """Compute layer states from oscillator field"""
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
        """Update goal orbital dynamics"""
        dt = 0.05
        
        for goal in self.goals:
            if goal['pinned']:
                continue
            
            # Orbital mechanics
            x, y = goal['position']
            dist = np.sqrt(x**2 + y**2)
            
            if dist > 0.1:
                # Gravity + orbital velocity
                gravity = 0.3 / (dist ** 2)
                vx = goal.get('vx', 0.0)
                vy = goal.get('vy', 0.0)
                
                vx -= gravity * x / dist * dt
                vy -= gravity * y / dist * dt
                vx += 0.1 * (-y / dist) * dt
                vy += 0.1 * (x / dist) * dt
                vx *= 0.98
                vy *= 0.98
                
                goal['vx'] = vx
                goal['vy'] = vy
                goal['position'] = [x + vx * dt, y + vy * dt]
            
            # Confidence fluctuation
            goal['confidence'] += (np.random.rand() - 0.5) * 0.02
            goal['confidence'] = np.clip(goal['confidence'], 0.1, 1.0)
        
        # Spawn/dissolve goals
        if len(self.goals) < 6 and np.random.rand() < 0.05:
            angle = np.random.rand() * 2 * np.pi
            radius = 0.6 + np.random.rand() * 0.3
            self.goals.append({
                'name': np.random.choice(['Discover', 'Optimize', 'Validate', 'Transform']),
                'confidence': 0.3 + np.random.rand() * 0.3,
                'position': [radius * np.cos(angle), radius * np.sin(angle)],
                'pinned': False,
                'hypotheses': [f'H{i}' for i in range(np.random.randint(1, 4))],
                'vx': 0.0,
                'vy': 0.0
            })
        
        self.goals = [g for g in self.goals if g['confidence'] > 0.15 or g['pinned']]
    
    def _update_sensors(self):
        """Update sensor values from field"""
        field = self.demo.oscillator.field
        self.sensors['visual'] = float(field[:len(field)//3].mean().item())
        self.sensors['temporal'] = float(field.var().item())
        self.sensors['pattern'] = float(np.clip(1.0 - field.std().item(), 0, 1))
    
    def _generate_actions(self):
        """Generate proposed actions"""
        # Remove old actions
        self.proposed_actions = [
            a for a in self.proposed_actions 
            if self.time_step - a.get('timestamp', 0) < 30
        ]
        
        # Generate new actions
        if np.random.rand() < 0.15 and len(self.proposed_actions) < 5:
            action_types = [
                ("Adjust Learning Rate", "Optimize training dynamics", 0.7),
                ("Explore New State", "Discover novel patterns", 0.6),
                ("Consolidate Memory", "Strengthen connections", 0.8),
                ("Query External", "Seek information", 0.5),
                ("Refine Hypothesis", "Improve accuracy", 0.75)
            ]
            
            name, outcome, base_mag = action_types[np.random.randint(0, len(action_types))]
            
            self.proposed_actions.append({
                'name': name,
                'magnitude': float(base_mag * self.sensors['pattern'] * np.random.rand()),
                'uncertainty': 1.0 - self.sensors['pattern'],
                'outcome_estimate': outcome,
                'approved': False,
                'timestamp': self.time_step
            })
    
    def get_cognition_snapshot(self) -> CognitionState:
        """Get current cognition state as a snapshot"""
        return CognitionState(
            timestamp=datetime.now().timestamp(),
            layers=self.layer_states,
            goals=self.goals,
            sensors=self.sensors,
            proposed_actions=self.proposed_actions,
            field_coherence=1.0 - float(self.demo.oscillator.field.std().item()),
            memory_magnitude=float(self.demo.oscillator.field.abs().mean().item())
        )
    
    async def broadcast_cognition(self):
        """Broadcast cognition state to all connected WebSocket clients"""
        if not self.websocket_clients:
            return
        
        snapshot = self.get_cognition_snapshot()
        message = {
            "type": "cognition_update",
            "data": snapshot.dict()
        }
        
        disconnected = []
        for client in self.websocket_clients:
            try:
                await client.send_json(message)
            except:
                disconnected.append(client)
        
        # Remove disconnected clients
        for client in disconnected:
            self.websocket_clients.remove(client)


# Global dashboard state
dashboard = DashboardState()


# ==================== Background Tasks ====================

async def cognition_loop():
    """Background task for continuous cognition updates"""
    while True:
        if dashboard.running:
            dashboard.update_cognition()
            await dashboard.broadcast_cognition()
        await asyncio.sleep(0.1)  # 10 Hz update rate


@app.on_event("startup")
async def startup_event():
    """Start background tasks on server startup"""
    asyncio.create_task(cognition_loop())
    logger.info("🚀 Neural Command Center backend started")
    logger.info("   Cognition engine: READY")
    logger.info("   WebSocket streaming: ENABLED")


# ==================== REST Endpoints ====================

@app.get("/")
async def root():
    """Health check"""
    return {
        "service": "CognitionSim Neural Command Center",
        "version": "2.0.0",
        "status": "online",
        "cognition_active": dashboard.running
    }


@app.get("/api/status")
async def get_status():
    """Get system status"""
    return {
        "cognition_running": dashboard.running,
        "training_active": dashboard.is_training,
        "connected_clients": len(dashboard.websocket_clients),
        "time_step": dashboard.time_step,
        "active_goals": len(dashboard.goals),
        "active_policies": len([p for p in dashboard.policies.values() if p.enabled]),
        "edge_devices": len(dashboard.edge_devices)
    }


@app.post("/api/cognition/start")
async def start_cognition():
    """Start cognition engine"""
    dashboard.running = True
    logger.info("▶ Cognition engine ACTIVATED")
    return {"status": "started", "message": "Cognition engine activated"}


@app.post("/api/cognition/stop")
async def stop_cognition():
    """Stop cognition engine"""
    dashboard.running = False
    logger.info("⏸ Cognition engine PAUSED")
    return {"status": "stopped", "message": "Cognition engine paused"}


@app.get("/api/cognition/snapshot")
async def get_cognition_snapshot():
    """Get current cognition state snapshot"""
    return dashboard.get_cognition_snapshot()


@app.post("/api/cognition/interact")
async def interact(command: InteractionCommand):
    """Send interaction command to cognition system"""
    logger.info(f"🎮 Interaction: {command.command_type} -> {command.target}")
    
    if command.command_type == "pin_goal":
        for goal in dashboard.goals:
            if goal['name'] == command.target:
                goal['pinned'] = True
                goal['confidence'] = 1.0
                return {"status": "success", "message": f"Goal '{command.target}' pinned"}
    
    elif command.command_type == "inject_constraint":
        # Inject constraint affects all goals with matching name
        for goal in dashboard.goals:
            if command.target.lower() in goal['name'].lower():
                goal['confidence'] *= 1.2
                goal['confidence'] = min(1.0, goal['confidence'])
        return {"status": "success", "message": f"Constraint '{command.target}' injected"}
    
    elif command.command_type == "approve_action":
        for action in dashboard.proposed_actions:
            if action['name'] == command.target:
                action['approved'] = True
                return {"status": "success", "message": f"Action '{command.target}' approved"}
    
    elif command.command_type == "probe":
        # Cognitive probe - return analysis
        probe_type = command.target
        if probe_type == "uncertainty":
            uncertain_areas = {
                layer: state for layer, state in dashboard.layer_states.items()
                if state['conflict'] > 0.5
            }
            return {"status": "success", "data": uncertain_areas}
        elif probe_type == "conflicts":
            conflicts = [
                goal for goal in dashboard.goals 
                if goal['confidence'] < 0.5
            ]
            return {"status": "success", "data": conflicts}
    
    return {"status": "unknown_command", "message": f"Unknown command type: {command.command_type}"}


# ==================== Governance Endpoints ====================

@app.get("/api/governance/policies")
async def get_policies():
    """Get all governance policies"""
    return {"policies": list(dashboard.policies.values())}


@app.post("/api/governance/policies")
async def create_policy(policy: GovernancePolicy):
    """Create new governance policy"""
    dashboard.policies[policy.policy_id] = policy
    logger.info(f"📋 Policy created: {policy.name}")
    return {"status": "created", "policy": policy}


@app.put("/api/governance/policies/{policy_id}")
async def update_policy(policy_id: str, policy: GovernancePolicy):
    """Update existing policy"""
    if policy_id not in dashboard.policies:
        raise HTTPException(status_code=404, detail="Policy not found")
    dashboard.policies[policy_id] = policy
    logger.info(f"📝 Policy updated: {policy.name}")
    return {"status": "updated", "policy": policy}


@app.delete("/api/governance/policies/{policy_id}")
async def delete_policy(policy_id: str):
    """Delete policy"""
    if policy_id not in dashboard.policies:
        raise HTTPException(status_code=404, detail="Policy not found")
    deleted = dashboard.policies.pop(policy_id)
    logger.info(f"🗑️  Policy deleted: {deleted.name}")
    return {"status": "deleted", "policy_id": policy_id}


# ==================== Training Endpoints ====================

@app.get("/api/training/metrics")
async def get_training_metrics():
    """Get recent training metrics"""
    return {"metrics": list(dashboard.training_metrics)}


@app.post("/api/training/metrics")
async def add_training_metric(metric: TrainingMetrics):
    """Add new training metric (for live dashboard integration)"""
    dashboard.training_metrics.append(metric.dict())
    return {"status": "recorded"}


# ==================== Edge Deployment Endpoints ====================

@app.get("/api/edge/devices")
async def get_edge_devices():
    """Get all edge device statuses"""
    return {"devices": list(dashboard.edge_devices.values())}


@app.post("/api/edge/devices")
async def register_edge_device(device: EdgeDeployment):
    """Register or update edge device"""
    dashboard.edge_devices[device.device_id] = device
    logger.info(f"📡 Edge device updated: {device.device_id} - {device.status}")
    return {"status": "registered", "device": device}


@app.get("/api/edge/devices/{device_id}")
async def get_edge_device(device_id: str):
    """Get specific edge device status"""
    if device_id not in dashboard.edge_devices:
        raise HTTPException(status_code=404, detail="Device not found")
    return dashboard.edge_devices[device_id]


# ==================== WebSocket Endpoints ====================

@app.websocket("/ws/cognition")
async def websocket_cognition(websocket: WebSocket):
    """WebSocket endpoint for real-time cognition streaming"""
    await websocket.accept()
    dashboard.websocket_clients.append(websocket)
    logger.info(f"🔌 WebSocket connected. Total clients: {len(dashboard.websocket_clients)}")
    
    try:
        while True:
            # Keep connection alive and listen for client messages
            data = await websocket.receive_text()
            # Echo back or handle client commands
            await websocket.send_json({"type": "ack", "message": "received"})
    except WebSocketDisconnect:
        dashboard.websocket_clients.remove(websocket)
        logger.info(f"🔌 WebSocket disconnected. Total clients: {len(dashboard.websocket_clients)}")


@app.websocket("/ws/training")
async def websocket_training(websocket: WebSocket):
    """WebSocket endpoint for real-time training metrics"""
    await websocket.accept()
    logger.info("🔌 Training WebSocket connected")
    
    try:
        while True:
            # Stream training metrics when available
            if dashboard.training_metrics:
                latest = dashboard.training_metrics[-1]
                await websocket.send_json({
                    "type": "training_update",
                    "data": latest
                })
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        logger.info("🔌 Training WebSocket disconnected")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
