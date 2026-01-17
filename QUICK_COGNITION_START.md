# QUICK START: See Cognition in Action

## The One-Command Demo

```bash
python demo_cognition.py
```

Then choose **option 7** for the complete demonstration.

**Total time: ~2 minutes**

---

## What You'll See

### 1. Neural Firing (15-20 seconds)
```
🧠 NEURAL SPIKING DEMONSTRATION

Iteration 1:
  🔴 Field Update Network: 42 neurons firing (16.80%)
  🟡 Syntropy Network:     38 neurons firing (15.20%)
  🟢 Feedback Network:     35 neurons firing (14.00%)
```

**What this means:** Neurons are firing at biologically plausible rates. The system is using sparse, efficient representations.

---

### 2. Field Coherence (15-20 seconds)
```
📡 FIELD COHERENCE DEMONSTRATION

Step 1:
  Field Mean:      0.0234
  Field Std Dev:   0.3421
  Coherence Score: 0.658 🔒 HIGH
```

**What this means:** The system's "mental state" is organized and synchronized. Higher coherence = more stable thinking.

---

### 3. Memory Learning (10-15 seconds)
```
💾 MEMORY CONSOLIDATION DEMONSTRATION

Experience 1:
  Memory Magnitude:        0.8234
  Consolidation Rate:      0.1000
  Integration Progress:    █░░░░░░░░░ 10.0%

Experience 2:
  Memory Magnitude:        0.7829
  Consolidation Rate:      0.9000
  Integration Progress:    █████████░ 90.0%
```

**What this means:** The system learns through exponential consolidation. New experiences blend with old memories smoothly.

---

### 4. Reasoning (10-15 seconds)
```
⚡ SYMBOLIC REASONING DEMONSTRATION

Reasoning Process 1:
  Input Concepts: ['intelligence', 'learning', 'adaptation']
  Inferred relationships: Central concept: intelligence, related: ['learning', 'adaptation']
```

**What this means:** The system builds knowledge graphs and discovers semantic relationships between concepts.

---

### 5. Integrated Processing (20-30 seconds)
```
🎯 INTEGRATED COGNITION DEMONSTRATION

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

**What this means:** All cognitive systems work together - neural processing → field evolution → memory consolidation → symbolic reasoning.

---

## After the Demo

### Option A: Run Individual Demonstrations
```bash
python demo_cognition.py
# Choose 1-5 to run individual components
```

### Option B: Extract Raw Data
```python
from demo_cognition import CognitionDemo

demo = CognitionDemo(field_size=100, device='cpu')
demo.observe_neural_spiking(num_iterations=5)

# Access observations
print(demo.observer.spike_history)
print(demo.observer.field_coherence_history)
print(demo.observer.memory_state_history)
print(demo.observer.reasoning_traces)
```

### Option C: Customize for Your Data
```python
from demo_cognition import CognitionDemo

demo = CognitionDemo(field_size=256)  # Larger field

# Process your own text
demo.symbolic_interpreter.build_knowledge_graph(
    concepts=["your", "concepts", "here"],
    neural_output=demo.oscillator.process_streamed_data("your text")
)
```

### Option D: Full System Training
```bash
# Quick training (2 min)
python train_quadra_matrix.py --epochs 20

# Full training (10 min)
python train_quadra_matrix.py --epochs 100
```

---

## Understanding the Metrics

### Neural Firing Rate: 15-20%
- **Meaning:** Neurons fire in sparse bursts, not continuously
- **Why:** Biologically plausible and computationally efficient
- **What to look for:** Consistent rates indicate healthy networks

### Field Coherence: 0.4-0.8
- **Meaning:** Global field organization (1.0 = perfect sync, 0.0 = chaos)
- **Why:** High coherence during rest, drops during processing, rises when solved
- **What to look for:** Natural rise-and-fall pattern indicates active thinking

### Memory Magnitude: Growing then Stable
- **Meaning:** Information stored in persistent memory
- **Why:** Exponential consolidation (new + old memories blend)
- **What to look for:** Should plateau after 5-10 experiences

### Reasoning Traces: Growing Knowledge
- **Meaning:** System builds increasingly complex knowledge graphs
- **Why:** Enables logical inference and concept relationship discovery
- **What to look for:** More concepts = deeper reasoning capacity

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "ImportError: quadra_matrix_spi" | Ensure `quadra_matrix_spi.py` exists in the root directory |
| Low firing rates (< 5%) | Normal! Spikes are sparse by design. Check they're not < 2% |
| Memory not growing | This is fine - consolidation maintains learned patterns |
| Slow execution | Use GPU: `device='cuda'` or reduce `field_size` |
| Demo crashes | Check PyTorch/snntorch/SymPy are installed: `pip install -r requirements.txt` |

---

## Key Insights

### 1. Sparse is Smart
Spikes should fire 15-20% of the time. This mimics biological brains and saves computation.

### 2. Coherence Tells a Story
- Rising coherence = system converging on a solution
- Falling coherence = system exploring possibilities
- High coherence throughout = system is stuck (no exploration)

### 3. Memory is Exponential
New experiences don't erase old memories - they blend together exponentially. The system keeps learning history.

### 4. Symbols + Neurons
The system combines:
- Neural networks (pattern learning)
- Symbolic reasoning (logical inference)
- This is not possible in pure neural or pure symbolic systems!

---

## Next: Deep Dive Guides

For more details on each component:

- 📖 [Full Cognition Observation Guide](./COGNITION_OBSERVATION_GUIDE.md) - Detailed explanation of each cognitive process
- 🧠 [Cognitive Model](./COGNITIVE_MODEL.md) - Mathematical foundations
- 🏗️ [Architecture](./ARCHITECTURE.md) - System design and components
- 📚 [Training Guide](./TRAINING.md) - How to train the full system

---

## Summary

✅ **Installation:** Included in main environment
✅ **First Run:** `python demo_cognition.py` → choose 7
✅ **Duration:** ~2 minutes
✅ **Customizable:** Modify field size, iterations, concepts
✅ **Integrable:** Use components in your own code

**Start now:** `python demo_cognition.py`
