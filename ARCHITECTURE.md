# CognitionSim SPI Architecture

## System Overview

CognitionSim implements a **Stateful Symbolic Predictive Interpreter (SPI)** that combines:

- **Symbolic reasoning** via pattern algebra and logical inference
- **Spiking neural dynamics** with threshold-based activation
- **Neuroplastic adaptation** across requests
- **Temporal oscillations** for phase-modulated responses
- **Governance-aware inference** with policy-conditioned output gating

This is fundamentally incompatible with stateless commodity ML. It's an **agent architecture**, not a prediction service.

> 💡 **Understanding the Language:** This document uses both poetic metaphors and mechanical specifications. For a complete dual-language reference that translates concepts like "field breathing" into precise mathematical equations, see [DUAL_LANGUAGE_GLOSSARY.md](./DUAL_LANGUAGE_GLOSSARY.md).

---

## Core Components

### 1. **Symbolic Layer** (`quadra/core/symbolic/`)

#### `SymbolicConfig`
```python
@dataclass
class SymbolicConfig:
    input_dim: int = 128         # Input feature space
    hidden_dim: int = 256        # Internal representation
    output_dim: int = 64         # Symbolic output
    num_layers: int = 3          # Stacking depth
```

#### `PatternModule`
Extracts and reconstructs patterns via bottleneck:
```
Input (128) → FC1 (256) → ReLU → FC2 (256) → ReLU → FC3 (128) → Output
```
Acts as **pattern codec** for lossy compression. Sparse patterns emerge.

---

### 2. **Neural Dynamics Layer** (`quadra/core/neural/`)

#### `SpikingFieldUpdateNN` 
Implements spiking field equations with thresholds:
```
potential(x) = sigmoid(FC(x))
spike = (potential > threshold)
output = spike * potential  [multiply-by-spike gating]
```

This is **not** standard ReLU. The spike gate is sparse and temporal.

#### `OscillatorySynapseTheory`
Phase-modulated synaptic transmission:
```
phase(t) = phase(t-1) + 0.1
output = tanh(FC(x)) * sin(phase(t))
```

Creates **temporal periodicity** — output magnitude oscillates at fixed frequency. Essential for:
- Circadian-like patterns
- Resonance phenomena
- Energy conservation

#### `SpikingSyntropyNN`
"Negative entropy" or order-promoting layer:
```
output = sigmoid(FC(x))
```
Drives the system toward **organized, low-entropy states** during inference.

#### `NeuroplasticityManager`
Exponential memory with plasticity:
```
memory(t) = 0.9 * memory(t-1) + 0.1 * adapted(t)
adapted(t) = tanh(FC(x))
```

**Critical property**: Memory persists across requests. Enables learning from interaction history.

#### `SpikingFeedbackNN`
Implements **recurrent feedback** from output to hidden state:
```
hidden' = hidden + 0.1 * feedback
output = ReLU(FC(hidden'))
```

Enables **attractor dynamics** and error correction.

---

### 3. **Governance Layer** (`quadra/core/governance/`)

#### `PolicyAdapter`
Maps system state → governance policy:
```python
class PolicyAdapter:
    def evaluate(self, context: InferenceContext) -> GovernancePolicy:
        """
        Given model state + input, determine:
        - Should output be suppressed?
        - Should explanation be attached?
        - Should prediction be deferred?
        """
        return policy
```

#### `GovernanceScorer`
Computes governance scores per output dimension:
```
score = policy_embedding @ output_representation
if score > threshold: output *= suppression_factor (e.g., 0.3)
```

---

### 4. **State Management** (`quadra/state/`)

#### `MemoryStore`
Persistent neuroplastic state:
```python
store.update_memory(session_id, new_state)
state = store.get_memory(session_id, age_decay=0.95)
```

Enables:
- **Cross-request learning** (memory persists)
- **Temporal decay** (older memories fade)
- **Session isolation** (per-user state)

---

## Data Flow: The Complete Pipeline

```
┌─────────────────┐
│ Raw Input (128) │
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│ 1. Pattern Encoding  │  PatternModule
│   Input → Bottleneck │  Extracts sparse features
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ 2. Field Spiking     │  SpikingFieldUpdateNN
│   threshold gating   │  Sparse temporal activation
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ 3. Syntropy         │  SpikingSyntropyNN
│   Order-promoting    │  Drive toward organized states
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ 4. Neuroplasticity   │  NeuroplasticityManager
│   Adaptive memory    │  Update self.memory (EMA)
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ 5. Oscillation       │  OscillatorySynapseTheory
│   Phase modulation   │  Temporal sinusoidal gating
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ 6. Feedback Loop     │  SpikingFeedbackNN
│   Recurrent update   │  Error correction, attractor
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ 7. Symbolic Reason   │  SymbolicPredictiveInterpreter
│   Pattern algebra    │  Extract logical structure
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ 8. Governance Gate   │  PolicyAdapter + GovernanceScorer
│   Policy-conditioned │  Suppress/explain as needed
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ Output (64 dims)     │  Policy-gated result
│ + Confidence         │  + Explanation if required
└──────────────────────┘
```

---

## State Persistence Architecture

### Session-Scoped State

```
Request 1: Input_1 → [Inference Pipeline] → NeuroplasticityManager.memory = M_1
                                           → MemoryStore[session_id] = M_1

Request 2: Input_2 → [Inference Pipeline] → NeuroplasticityManager.memory = 0.9*M_1 + 0.1*adapted_2
                                           → MemoryStore[session_id] = M_2
                     (Memory "remembers" Request 1's effect)

Request 3: Input_3 → [Same pipeline] → Memory blended from Requests 1 & 2
```

**Result**: Model **adapts** to session-specific patterns. Not stateless batch processing.

---

## Governance-Inference Integration

### Policy-Conditioned Output

```python
# After inference pipeline produces `output`:

policy = governance_adapter.evaluate(context)

# Gate 1: Suppression
if policy.requires_suppression:
    output *= 0.3  # Confidence reduction

# Gate 2: Explanation
if policy.requires_explanation:
    # Attach symbolic trace showing decision path
    output.explanation = symbolic_trace

# Gate 3: Escalation
if policy.requires_escalation:
    # Route to human reviewer
    defer_prediction(output)
    return None
```

**Key insight**: Governance is **not post-hoc moderation**. It's **embedded in inference**. The model learns that certain outputs get gated, shaping learned behavior.

---

## Comparison: Commodity ML vs. SPI Architecture

| Aspect | Commodity ML | SPI |
|--------|-------------|-----|
| **State** | Stateless batch processing | Persistent memory per session |
| **Memory** | Input-output pairs | Internal neuroplastic state |
| **Governance** | Post-hoc filtering | Policy-conditioned inference |
| **Adaptation** | Requires retraining | Online learning via memory |
| **Temporal** | None (iid assumption) | Phase-modulated, oscillatory |
| **Feedback** | Supervised labels | Self-correcting recurrent loops |
| **Architecture** | Feed-forward | Recurrent + memory + governance |

---

## Deployment Topology

```
┌────────────────────────────────────────┐
│           Flask API Gateway            │
│  (TLS, Auth, Request Routing)          │
└──────────────┬─────────────────────────┘
               │
               ▼
┌────────────────────────────────────────┐
│     Inference Service (replicated)     │
│  • SymbolicPredictiveInterpreter       │
│  • PolicyAdapter                       │
└──────────────┬─────────────────────────┘
               │
               ▼
┌────────────────────────────────────────┐
│   Stateful Memory Store (redis/db)     │
│  • MemoryStore[session_id] → state     │
│  • TTL-based expiry                    │
└────────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────┐
│   Governance Policy Engine             │
│  • PolicyAdapter                       │
│  • SocialGovernanceService             │
│  • Audit log (all decisions)           │
└────────────────────────────────────────┘
```

---

## Critical Design Decisions

### 1. **Multiplicative Gating, Not Additive**
- Output *= gating_factor (preserves structure)
- Not output += correction (can destroy learned patterns)

### 2. **Exponential Memory Decay**
- memory(t) = 0.9 * memory(t-1) + 0.1 * new
- Forgets old patterns gracefully
- Not hard reset (preserves adaptation)

### 3. **Spiking > ReLU**
- Spike gate: (potential > threshold).float()
- ReLU: max(0, x)
- Spikes are **sparse, temporal, meaningful**

### 4. **Policy as First-Class**
- Not filtering output_post_inference
- Policy shapes network behavior during forward pass
- Enables **learned compliance**

---

## Next Steps for Production

1. **Load persistent state on startup** → MemoryStore from checkpoint
2. **Governance policy versioning** → Track policy changes for audit
3. **Monitoring metrics** → Track memory staleness, policy enforcement rate
4. **Failure recovery** → If MemoryStore unavailable, degrade to stateless mode
5. **Multi-agent communication** → Oscillatory synchronization between instances
