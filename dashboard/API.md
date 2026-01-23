# Neural Command Center API Reference

## Overview

The Neural Command Center provides a REST API and WebSocket endpoints for managing cognitive processes, training, governance, and edge deployment.

**Base URL**: `http://localhost:8000`  
**API Docs**: `http://localhost:8000/docs` (Swagger UI)  
**ReDoc**: `http://localhost:8000/redoc`

---

## Authentication

**Current**: No authentication required  
**Future**: JWT Bearer tokens

```http
Authorization: Bearer <token>
```

---

## REST API

### Health & Status

#### `GET /`

Health check endpoint

**Response**:
```json
{
  "status": "healthy",
  "message": "Neural Command Center API"
}
```

#### `GET /api/status`

Get system status

**Response**:
```json
{
  "cognition_running": true,
  "training_active": false,
  "edge_devices": 3,
  "policies_active": 5,
  "uptime": 3600
}
```

---

### Cognition Endpoints

#### `POST /api/cognition/start`

Start the cognition engine

**Request Body**:
```json
{
  "field_size": 100,
  "update_interval": 0.1
}
```

**Response**:
```json
{
  "status": "started",
  "message": "Cognition engine started"
}
```

#### `POST /api/cognition/stop`

Stop the cognition engine

**Response**:
```json
{
  "status": "stopped",
  "message": "Cognition engine stopped"
}
```

#### `GET /api/cognition/snapshot`

Get current cognitive state

**Response**:
```json
{
  "field": [[...]],  // 100x100 array
  "attention": [0.5, 0.5],
  "uncertainty": 0.3,
  "conflicts": [
    {"layer": "reasoning", "intensity": 0.8}
  ],
  "goals": [
    {
      "id": "goal_1",
      "description": "Maximize accuracy",
      "priority": 0.9,
      "hypotheses": [
        {
          "description": "Use ensemble",
          "confidence": 0.7
        }
      ]
    }
  ],
  "sensors": {
    "visual": {"value": 0.5, "quality": 0.9},
    "temporal": {"value": 0.3, "quality": 0.85},
    "pattern": {"value": 0.7, "quality": 0.95}
  },
  "proposed_actions": [
    {
      "id": "action_1",
      "description": "Adjust learning rate",
      "vector": [0.1, -0.2, 0.5],
      "uncertainty": 0.2
    }
  ],
  "timestamp": 1234567890.123
}
```

#### `POST /api/cognition/interact`

Send interaction command to cognition engine

**Request Body**:
```json
{
  "type": "signal_injection",
  "data": {
    "sensor": "visual",
    "value": 0.8
  }
}
```

**Alternative Types**:

**Cognitive Probe**:
```json
{
  "type": "probe",
  "data": {
    "query": "uncertainty",
    "layer": "reasoning"
  }
}
```

**Safety Governor**:
```json
{
  "type": "safety",
  "data": {
    "constraint": "max_action_magnitude",
    "value": 0.5
  }
}
```

**Goal Pinning**:
```json
{
  "type": "pin_goal",
  "data": {
    "goal_id": "goal_1",
    "pinned": true
  }
}
```

**Action Approval**:
```json
{
  "type": "approve_action",
  "data": {
    "action_id": "action_1",
    "approved": true
  }
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Interaction processed",
  "result": {...}
}
```

---

### Governance Endpoints

#### `GET /api/governance/policies`

List all policies

**Response**:
```json
{
  "policies": [
    {
      "id": "policy_1",
      "name": "Max Inference Time",
      "type": "hard",
      "enabled": true,
      "parameters": {
        "max_time_ms": 100
      }
    },
    {
      "id": "policy_2",
      "name": "Prefer Accuracy",
      "type": "soft",
      "enabled": true,
      "parameters": {
        "weight": 0.8
      }
    }
  ]
}
```

#### `POST /api/governance/policies`

Create a new policy

**Request Body**:
```json
{
  "name": "Energy Budget",
  "type": "contextual",
  "enabled": true,
  "parameters": {
    "max_watts": 50,
    "context": "battery_mode"
  }
}
```

**Response**:
```json
{
  "id": "policy_3",
  "status": "created",
  "policy": {...}
}
```

#### `PUT /api/governance/policies/{policy_id}`

Update an existing policy

**Request Body**:
```json
{
  "enabled": false
}
```

**Response**:
```json
{
  "status": "updated",
  "policy": {...}
}
```

#### `DELETE /api/governance/policies/{policy_id}`

Delete a policy

**Response**:
```json
{
  "status": "deleted",
  "id": "policy_3"
}
```

---

### Training Endpoints

#### `GET /api/training/metrics`

Get training metrics

**Query Parameters**:
- `limit` (optional): Number of recent metrics to return (default: 100)

**Response**:
```json
{
  "metrics": [
    {
      "epoch": 1,
      "step": 100,
      "loss": 2.34,
      "accuracy": 0.67,
      "learning_rate": 0.001,
      "timestamp": 1234567890.123
    },
    ...
  ]
}
```

#### `POST /api/training/metrics`

Add training metric (for integration with `train_live_dashboard.py`)

**Request Body**:
```json
{
  "epoch": 1,
  "step": 100,
  "loss": 2.34,
  "accuracy": 0.67,
  "learning_rate": 0.001
}
```

**Response**:
```json
{
  "status": "added",
  "metric": {...}
}
```

---

### Edge Deployment Endpoints

#### `GET /api/edge/devices`

List all edge devices

**Response**:
```json
{
  "devices": [
    {
      "id": "device_1",
      "name": "Raspberry Pi 4",
      "status": "online",
      "last_seen": 1234567890.123,
      "metrics": {
        "latency_ms": 45,
        "memory_mb": 512,
        "cpu_percent": 30,
        "inferences_per_sec": 10
      }
    },
    ...
  ]
}
```

#### `POST /api/edge/devices`

Register or update edge device

**Request Body**:
```json
{
  "id": "device_2",
  "name": "Jetson Nano",
  "status": "online",
  "metrics": {
    "latency_ms": 20,
    "memory_mb": 1024,
    "cpu_percent": 50,
    "inferences_per_sec": 30
  }
}
```

**Response**:
```json
{
  "status": "registered",
  "device": {...}
}
```

#### `GET /api/edge/devices/{device_id}`

Get specific device status

**Response**:
```json
{
  "id": "device_1",
  "name": "Raspberry Pi 4",
  "status": "online",
  "last_seen": 1234567890.123,
  "metrics": {...},
  "history": [...]
}
```

---

## WebSocket API

### Cognition Stream

#### `WS /ws/cognition`

Real-time cognition state stream

**Connection**:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/cognition');
```

**Message Format**:
```json
{
  "type": "cognition_update",
  "data": {
    "field": [[...]],
    "attention": [0.5, 0.5],
    "uncertainty": 0.3,
    "conflicts": [...],
    "goals": [...],
    "sensors": {...},
    "proposed_actions": [...]
  },
  "timestamp": 1234567890.123
}
```

**Update Rate**: ~10 Hz (100ms intervals)

**Client Example**:
```javascript
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  if (message.type === 'cognition_update') {
    updateVisualization(message.data);
  }
};
```

### Training Stream

#### `WS /ws/training` (Future)

Real-time training metrics stream

**Message Format**:
```json
{
  "type": "training_update",
  "data": {
    "epoch": 1,
    "step": 100,
    "loss": 2.34,
    "accuracy": 0.67
  },
  "timestamp": 1234567890.123
}
```

---

## Error Responses

All endpoints use standard HTTP status codes:

**Success**: 200 OK
```json
{
  "status": "success",
  "data": {...}
}
```

**Bad Request**: 400
```json
{
  "detail": "Invalid field_size: must be > 0"
}
```

**Not Found**: 404
```json
{
  "detail": "Policy not found: policy_99"
}
```

**Server Error**: 500
```json
{
  "detail": "Internal server error"
}
```

---

## Rate Limits

**Current**: None  
**Future**: 
- Anonymous: 100 req/min
- Authenticated: 1000 req/min
- WebSocket: 10 connections per IP

---

## Integration Examples

### Python (requests)

```python
import requests

# Start cognition
response = requests.post('http://localhost:8000/api/cognition/start', json={
    'field_size': 100,
    'update_interval': 0.1
})
print(response.json())

# Get snapshot
snapshot = requests.get('http://localhost:8000/api/cognition/snapshot').json()
print(f"Uncertainty: {snapshot['uncertainty']}")

# Send interaction
requests.post('http://localhost:8000/api/cognition/interact', json={
    'type': 'signal_injection',
    'data': {'sensor': 'visual', 'value': 0.8}
})
```

### Python (WebSocket)

```python
import asyncio
import websockets
import json

async def watch_cognition():
    async with websockets.connect('ws://localhost:8000/ws/cognition') as ws:
        async for message in ws:
            data = json.loads(message)
            print(f"Uncertainty: {data['data']['uncertainty']}")

asyncio.run(watch_cognition())
```

### JavaScript (fetch)

```javascript
// Start cognition
const response = await fetch('http://localhost:8000/api/cognition/start', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({field_size: 100, update_interval: 0.1})
});
const result = await response.json();
console.log(result);

// Get snapshot
const snapshot = await fetch('http://localhost:8000/api/cognition/snapshot')
  .then(r => r.json());
console.log(`Uncertainty: ${snapshot.uncertainty}`);
```

### JavaScript (WebSocket)

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/cognition');

ws.onopen = () => console.log('Connected');
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  console.log(`Uncertainty: ${msg.data.uncertainty}`);
};
ws.onerror = (error) => console.error('WebSocket error:', error);
ws.onclose = () => console.log('Disconnected');
```

### cURL

```bash
# Health check
curl http://localhost:8000/

# Start cognition
curl -X POST http://localhost:8000/api/cognition/start \
  -H "Content-Type: application/json" \
  -d '{"field_size": 100, "update_interval": 0.1}'

# Get snapshot
curl http://localhost:8000/api/cognition/snapshot

# Create policy
curl -X POST http://localhost:8000/api/governance/policies \
  -H "Content-Type: application/json" \
  -d '{"name": "Test Policy", "type": "hard", "enabled": true}'

# List policies
curl http://localhost:8000/api/governance/policies
```

---

## Changelog

**v1.0.0** (2025-01-XX)
- Initial release
- REST API for cognition, governance, training, edge
- WebSocket streaming for cognition
- Pydantic validation
- OpenAPI documentation

**Future Versions**
- v1.1.0: Authentication (JWT)
- v1.2.0: Rate limiting
- v1.3.0: Webhooks for events
- v2.0.0: GraphQL endpoint

---

**API Version**: 1.0.0  
**Last Updated**: 2025-01-XX  
**Maintainer**: CognitionSim Team
