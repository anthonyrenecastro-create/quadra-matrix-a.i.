# How to See and Understand Quadra Matrix Cognition

## TL;DR - Quick Start (2 minutes)

```bash
# Run the interactive cognition demo
python demo_cognition.py

# Then choose option 7 for the "Full Sequence"
# This demonstrates all cognitive processes
```

---

## Overview

Quadra Matrix implements a **hybrid cognitive architecture** combining:
- **Spiking Neural Networks** - Biologically-inspired sparse firing
- **Neuroplastic Memory** - Exponential consolidation over time
- **Symbolic Reasoning** - Logical inference and knowledge graphs
- **Oscillatory Dynamics** - Field coherence and resonance

This guide shows you how to observe and understand each component.

---

## 1. Neural Spiking - Watch Neurons Fire

### What You're Observing

Spiking neural networks are fundamentally different from traditional neural networks. Instead of continuous activations, neurons either **fire (emit a spike)** or **rest (silent)**.

**Mathematical Model:**
```
potential(t) = sigmoid(W·x(t) + b)
spike(t) = [potential(t) > θ]  ∈ {0, 1}    ← threshold gate
output(t) = spike(t) * potential(t)        ← gated output
```

**Why This Matters:**
- Sparse: Most neurons are silent (efficiency)
- Temporal: Spikes are discrete events in time (interpretability)
- Biological: Mimics real neural firing (plausibility)

### How to Observe It

**Interactive Demo:**
```bash
python demo_cognition.py
# Choose option 1: "Neural Spiking - Watch neurons fire and spike"
```

**What You'll See:**

```
Iteration 1:
  🔴 Field Update Network: 42 neurons firing (16.80%)
  🟡 Syntropy Network:     38 neurons firing (15.20%)
  🟢 Feedback Network:     35 neurons firing (14.00%)

Iteration 2:
  🔴 Field Update Network: 45 neurons firing (18.00%)
  🟡 Syntropy Network:     40 neurons firing (16.00%)
  🟢 Feedback Network:     37 neurons firing (14.80%)
```

### Interpretation Guide

| Metric | Meaning | What It Tells You |
|--------|---------|-------------------|
| **Neurons Firing** | Number of active neurons in each cycle | Network activity level |
| **Firing Rate (%)** | Percentage of neurons that fired | Sparsity/efficiency |
| 15-20% | Normal range | Efficient, biologically plausible |
| < 10% | Very sparse | Over-silencing, possible dead neurons |
| > 30% | Dense firing | High activity, possible overstimulation |

### Code Example

```python
from demo_cognition import CognitionDemo
import torch

demo = CognitionDemo(field_size=100, device='cpu')

# Observe 5 iterations of neural spiking
demo.observe_neural_spiking(num_iterations=5)

# Access raw spike data
spikes = demo.observer.spike_history
for record in spikes:
    print(f"Active neurons: {record['active_neurons']}, Firing rate: {record['firing_rate_percent']:.2f}%")
```

---

## 2. Field Coherence - Track Field State Stability

### What You're Observing

The field state represents the system's overall "mental state" - like the collective synchronization of all neurons. **Coherence** measures how organized and stable this state is.

**Mathematical Properties:**
```
Coherence = 1.0 - min(1.0, std(field_state))
Coherence ∈ [0, 1]:
  - 1.0 = Perfect coherence (all neurons synchronized)
  - 0.5 = Medium coherence (partial synchronization)
  - 0.0 = Low coherence (completely scattered activation)
```

### How to Observe It

**Interactive Demo:**
```bash
python demo_cognition.py
# Choose option 2: "Field Coherence - Track field state and stability"
```

**What You'll See:**

```
Step 1:
  Field Mean:      0.0234
  Field Std Dev:   0.3421
  Coherence Score: 0.658 🔒 HIGH

Step 2:
  Field Mean:      0.0156
  Field Std Dev:   0.5123
  Coherence Score: 0.488 ❓ MEDIUM

Step 3:
  Field Mean:      0.0089
  Field Std Dev:   0.7234
  Coherence Score: 0.277 🌪️  LOW
```

### Interpretation Guide

| Coherence | Pattern | Meaning |
|-----------|---------|---------|
| **0.7 - 1.0** 🔒 | Organized, synchronized | System is stable and focused |
| **0.4 - 0.7** ❓ | Partially organized | System in transition or processing |
| **0.0 - 0.4** 🌪️ | Chaotic, scattered | System is confused or overloaded |

### What Happens During Real Cognitive Tasks

1. **At Rest**: High coherence (0.8+) - System is settled
2. **Processing Input**: Coherence drops to 0.4-0.6 - System explores patterns
3. **Finding Solution**: Coherence rises to 0.7+ - System converges
4. **Next Task**: Cycle repeats

### Code Example

```python
demo = CognitionDemo(field_size=100)
demo.observe_field_coherence(num_steps=5)

# Analyze coherence evolution
coherence_data = demo.observer.field_coherence_history
import statistics
avg_coherence = statistics.mean(x['coherence'] for x in coherence_data)
print(f"Average coherence: {avg_coherence:.3f}")
```

---

## 3. Memory Consolidation - See Learning Happen

### What You're Observing

**Neuroplasticity** is how the system learns. New experiences are consolidated into persistent memory through **exponential moving average (EMA)**:

```
M(t) = α·M(t-1) + (1-α)·X(t)

Where:
- M(t-1) = old memory
- X(t) = new experience
- α = 0.9 (slow consolidation, long-term retention)
- (1-α) = 0.1 (gradual integration)
```

### How to Observe It

**Interactive Demo:**
```bash
python demo_cognition.py
# Choose option 3: "Memory Consolidation - See exponential memory formation"
```

**What You'll See:**

```
Experience 1:
  Memory Magnitude:        0.8234
  Consolidation Rate:      0.1000
  Memory Dimensionality:   100
  Integration Progress:    █░░░░░░░░░ 10.0%

Experience 2:
  Memory Magnitude:        0.7829
  Consolidation Rate:      0.9000
  Memory Dimensionality:   100
  Integration Progress:    █████████░ 90.0%

Experience 3:
  Memory Magnitude:        0.7541
  Consolidation Rate:      0.9000
  Memory Dimensionality:   100
  Integration Progress:    █████████░ 90.0%
```

### Interpretation Guide

| Property | Meaning |
|----------|---------|
| **Memory Magnitude** | Amount of information stored (grows initially, stabilizes) |
| **Consolidation Rate** | How much old memory vs new (0.9 = 90% old, 10% new) |
| **Integration Progress** | Visual representation of memory blend |

### Key Insight: Exponential Consolidation

The magic of neuroplasticity is **exponential consolidation**:

1. **First experience**: Memory = new experience
2. **Second experience**: Memory = 90% old + 10% new (keeps learning history)
3. **Third experience**: Memory = 90% blended + 10% newest
4. **Asymptotic behavior**: Memory stabilizes but never completely forgets

This is why the system can:
- ✓ Learn quickly from new patterns
- ✓ Maintain long-term knowledge
- ✓ Gracefully transition between tasks

### Code Example

```python
demo = CognitionDemo(field_size=100)
demo.observe_memory_consolidation()

# Extract memory evolution
memory_data = demo.observer.memory_state_history
for i, record in enumerate(memory_data):
    print(f"Step {i+1}: Magnitude={record['magnitude']:.4f}, Rate={record['consolidation_rate']:.1%}")
```

---

## 4. Symbolic Reasoning - Watch Logical Inference

### What You're Observing

The system doesn't just learn patterns - it also **builds knowledge graphs** and performs **symbolic inference**:

```
Input Concepts → Knowledge Graph Construction → Logical Inference
```

### How to Observe It

**Interactive Demo:**
```bash
python demo_cognition.py
# Choose option 4: "Symbolic Reasoning - Observe logical inference"
```

**What You'll See:**

```
Reasoning Process 1:
  Input Concepts: ['intelligence', 'learning', 'adaptation']
  Inferred relationships: Central concept: intelligence, related: ['learning', 'adaptation']

Reasoning Process 2:
  Input Concepts: ['neural', 'network', 'computation']
  Inferred relationships: Central concept: network, related: ['neural', 'computation']

Reasoning Process 3:
  Input Concepts: ['memory', 'consolidation', 'recall']
  Inferred relationships: Central concept: consolidation, related: ['memory', 'recall']
```

### What's Happening Under the Hood

1. **Tokenization**: Break concepts into words
2. **Synonym Discovery**: Use WordNet to find related terms
3. **Graph Construction**: Build NetworkX knowledge graph
4. **Relationship Inference**: Identify central concepts and connections
5. **Semantic Analysis**: Link neural outputs to symbolic structures

### Interpretation Guide

| Element | Meaning |
|---------|---------|
| **Input Concepts** | The ideas you're asking the system to reason about |
| **Central Concept** | The most connected/important idea in the graph |
| **Related Concepts** | Other concepts connected through synonymy or semantic similarity |

### Code Example

```python
demo = CognitionDemo(field_size=100)
demo.observe_symbolic_reasoning()

# Access reasoning traces
traces = demo.observer.reasoning_traces
for trace in traces:
    print(f"Query: {trace['query']}")
    print(f"Concepts: {trace['concepts']}")
    print(f"Result: {trace['result']}\n")
```

---

## 5. Integrated Cognition - Everything Together

### What You're Observing

All four cognitive processes working in concert:

```
Text Input
    ↓
🧠 Neural Processing (spikes fire)
    ↓
📡 Field Evolution (coherence changes)
    ↓
💾 Memory Update (consolidation)
    ↓
⚡ Symbolic Reasoning (knowledge built)
    ↓
Output
```

### How to Observe It

**Interactive Demo:**
```bash
python demo_cognition.py
# Choose option 7: "Full Sequence - Run everything in order"
```

**Or Run Programmatically:**
```python
demo = CognitionDemo(field_size=100)
demo.observe_integrated_cognition()
```

**What You'll See:**

```
Cognitive Process 1: Processing 'learning and adaptation'

  Step 1️⃣  - Neural Processing
         Neurons activated: 42

  Step 2️⃣  - Field Evolution
         Field coherence: 0.658

  Step 3️⃣  - Memory Update
         Memory magnitude: 0.8234

  Step 4️⃣  - Symbolic Reasoning
         Concepts analyzed: 2
```

### Understanding the Full Cycle

| Step | Component | Purpose |
|------|-----------|---------|
| 1️⃣ | Neural Processing | Convert input to sparse representation |
| 2️⃣ | Field Evolution | Establish global coherence |
| 3️⃣ | Memory Update | Consolidate into long-term storage |
| 4️⃣ | Symbolic Reasoning | Build/update knowledge graph |

---

## 6. Performance Metrics - What to Look For

### Create a Monitoring Script

```python
from demo_cognition import CognitionDemo, CognitionObserver
import statistics

demo = CognitionDemo(field_size=100)

# Run a full sequence
demo.observe_integrated_cognition()

# Print detailed metrics
observer = demo.observer

print("=== NEURAL ACTIVITY ===")
if observer.spike_history:
    avg_spikes = statistics.mean(x['spike_count'] for x in observer.spike_history)
    print(f"Average spikes: {avg_spikes:.1f}")
    print(f"Peak firing rate: {max(x['firing_rate_percent'] for x in observer.spike_history):.2f}%")

print("\n=== FIELD STABILITY ===")
if observer.field_coherence_history:
    coherence = [x['coherence'] for x in observer.field_coherence_history]
    print(f"Average coherence: {statistics.mean(coherence):.3f}")
    print(f"Coherence variance: {statistics.variance(coherence):.4f}")

print("\n=== MEMORY CONSOLIDATION ===")
if observer.memory_state_history:
    memories = [x['magnitude'] for x in observer.memory_state_history]
    print(f"Initial memory: {memories[0]:.4f}")
    print(f"Final memory: {memories[-1]:.4f}")
    print(f"Consolidation trend: {'increasing' if memories[-1] > memories[0] else 'decreasing'}")

print("\n=== REASONING DEPTH ===")
if observer.reasoning_traces:
    avg_concepts = statistics.mean(x['num_concepts'] for x in observer.reasoning_traces)
    print(f"Average concepts per reasoning: {avg_concepts:.1f}")
    print(f"Total reasoning processes: {len(observer.reasoning_traces)}")
```

---

## 7. Troubleshooting & Common Questions

### Q: Why are my firing rates so low (< 5%)?
**A:** This is actually good! Spikes are meant to be sparse. If they're extremely low (< 2%), the threshold might be too high. Check the spiking networks are initialized correctly.

### Q: Field coherence keeps dropping - is something wrong?
**A:** No - during active processing, coherence naturally drops. This is how the system "explores" solution space. High coherence throughout indicates the system isn't exploring enough.

### Q: Memory magnitude isn't growing - is learning happening?
**A:** Memory magnitude can fluctuate. What matters is the **consolidation pattern**. As long as consolidation rate stays around 0.9, learning is happening correctly.

### Q: How do I run this on GPU?
**A:** Automatically detected:
```python
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
demo = CognitionDemo(field_size=100, device=device)
```

### Q: Can I customize the demonstrations?
**A:** Yes! Modify `demo_cognition.py`:
```python
# Change field size
demo = CognitionDemo(field_size=256)  # Larger field = more neurons

# Change number of iterations
demo.observe_neural_spiking(num_iterations=10)  # More iterations

# Add your own text
demo.observe_integrated_cognition()  # Modify test_inputs list
```

---

## 8. Next Steps - Deeper Exploration

### Option A: Study the Code
```bash
# Core cognitive components
cat quadra_matrix_spi.py | head -200

# Advanced features
cat COGNITIVE_MODEL.md | head -100

# Architecture details
cat ARCHITECTURE.md | head -150
```

### Option B: Run Full Training
```bash
# Quick demo
python train_quadra_matrix.py --epochs 20

# Full training
python train_quadra_matrix.py --epochs 100
```

### Option C: Integrate with Your Application
```python
from quadra_matrix_spi import OscillatorySynapseTheory
import torch

# Initialize the core engine
oscillator = OscillatorySynapseTheory(field_size=100, device='cpu')

# Process your data
text = "your text here"
features = oscillator.process_streamed_data(text)

# Get predictions
predictions = oscillator.predict(features)
```

---

## Summary

You now know how to observe and understand Quadra Matrix cognition:

- 🧠 **Neural Spiking**: 15-20% firing rate is normal and efficient
- 📡 **Field Coherence**: Watch it drop during processing, rise when done
- 💾 **Memory Consolidation**: Exponential EMA maintains learning history
- ⚡ **Symbolic Reasoning**: Knowledge graphs build incrementally
- 🎯 **Integrated Cognition**: All systems work together seamlessly

**Start exploring:** `python demo_cognition.py` and choose option 7!
