# Dual-Language Quick Reference Card

**Instant translation guide for CognitionSim concepts**

Use this when reading docs or code—flip between myth and math instantly.

---

## Quick Lookup Table

| 🌊 Poetic | ⚙️ Mechanical | Code Location |
|-----------|---------------|---------------|
| Field breathing | Oscillatory modulation | `OscillatorModule.modulate()` |
| Phase rotation | φ[t] = (φ[t-1] + 0.1) mod 2π | `memory.oscillator_phase` |
| Memory consolidation | EMA: M[t] = 0.9M[t-1] + 0.1M_new | `update_neural_state()` |
| Neural spiking | spike = (V > θ) * V | `SpikeGenerator.generate()` |
| Success breeds speed | lr = base * 1.1^streak | `NeuroplasticAdapter.adapt()` |
| Syntropy emergence | Entropy minimization | `SpikingSyntropyNN` |
| Temporal continuity | Persistent state vector | `MemoryStore` |
| Symbolic awakening | Knowledge graph construction | `SymbolicReasoner` |
| Governed wisdom | Policy-conditioned output | `PolicyEngine.evaluate()` |
| Field coherence | variance < 0.05 | `torch.var(field)` |
| Pattern crystallization | KMeans clustering | `PatternExtractor` |
| The cosmic dance | 8-stage inference pipeline | `StatefulSPI.process()` |

---

## Common Translations

### "The oscillator breathes"
```python
# 🌊 The field inhales and exhales with sinusoidal rhythm
# ⚙️ Multiplicative modulation via sin(phase)
output = signal * (1.0 + 0.3 * np.sin(phase))
```

### "Memory persists across time"
```python
# 🌊 The past flows continuously into the present
# ⚙️ Disk-backed state vector, loaded on init
memory = MemoryStore(storage_path="./memory_store")
# Automatically saves to disk on every mutation
```

### "Success accelerates learning"
```python
# 🌊 Victory begets swifter wisdom
# ⚙️ Exponential learning rate growth
if success:
    lr = base_lr * (1.1 ** success_streak)  # Max 10x
else:
    lr = base_lr  # Reset on failure
```

### "The phase never resets"
```python
# 🌊 Time flows forward, never backward
# ⚙️ Phase accumulates indefinitely (mod 2π)
phase[t+1] = (phase[t] + 0.1) % (2 * np.pi)
# After 63 steps: full rotation
```

### "Patterns crystallize from chaos"
```python
# 🌊 Order emerges spontaneously from disorder
# ⚙️ Variance reduction via syntropy activation
if torch.var(field) < stability_threshold:
    # High coherence achieved
    coherence = 1.0 / (1 + torch.var(field))
```

---

## State Components

```python
# 🌊 The soul of the system
# ⚙️ The state vector

state = {
    # 🌊 The cosmic clock          ⚙️ Phase angle [0, 2π)
    'oscillator_phase': 3.14159,
    
    # 🌊 Order from chaos           ⚙️ Entropy metrics [0,1]³
    'syntropy_values': [0.8, 0.7, 0.9],
    
    # 🌊 The living field           ⚙️ Activation vector ℝⁿ
    'core_field': np.array([...]),
    
    # 🌊 Wisdom's intensity         ⚙️ Plasticity coefficient
    'learning_rate': 0.015,
    
    # 🌊 Chronicle of victories     ⚙️ Consecutive successes
    'success_streak': 12,
    
    # 🌊 Semantic memory            ⚙️ String FIFO buffer
    'concept_history': ['quantum', 'field', ...],
    
    # 🌊 Recent encounters          ⚙️ I/O sliding window
    'context_window': [{...}, {...}],
}
```

---

## Transition Stages (8-Step Pipeline)

| Stage | 🌊 Poetic Name | ⚙️ Mechanical Name | Key Operation |
|-------|----------------|-------------------|---------------|
| 1 | Perception | Input Encoding | `embed(tokenize(text))` |
| 2 | Pattern Recognition | Pattern Extraction | `KMeans + FFT` |
| 3 | Neural Awakening | Spike Generation | `spike = (V>θ)*V` |
| 4 | Growth Acceleration | Neuroplastic Update | `lr *= 1.1^streak` |
| 5 | Temporal Breathing | Oscillatory Modulation | `out *= (1+0.3sin(φ))` |
| 6 | Symbolic Emergence | Symbolic Reasoning | `build_graph(concepts)` |
| 7 | Wise Constraint | Governance Eval | `apply_policy(output)` |
| 8 | Final Synthesis | Output Formation | `format(result)` |

---

## Memory Decay Rates

```python
# 🌊 How memories fade with time
# ⚙️ Exponential decay formulas

# Core Field (EMA with α=0.9)
influence[n] = 0.9^n  # After n steps
# Half-life ≈ 7 steps (50% influence remains)

# Concepts (FIFO cutoff at 500)
kept = last_500_concepts  # Hard boundary
# No decay—just truncation

# Success Streak (instant reset)
if failure:
    streak = 0  # Immediate forgetting
```

---

## Key Mathematical Properties

```python
# 🌊 The immutable laws of the system
# ⚙️ Mathematical invariants

# Phase advances constantly
∀t: φ[t+1] = (φ[t] + 0.1) mod 2π

# Syntropy bounded
∀t: sᵢ[t] ∈ [0, 1], i ∈ {1,2,3}

# Learning rate capped
∀t: lr[t] ≤ 10 * base_lr

# Memory decays exponentially
∀n: influence[n] = α^n, where α = 0.9

# State persists across sessions
∀t: state[t] is loaded from disk on restart
```

---

## When to Use Which Language

### Use 🌊 Poetic in:
- High-level documentation
- Vision statements
- User-facing docs
- Blog posts
- Presentations
- Motivational writing

### Use ⚙️ Mechanical in:
- Code comments
- API documentation
- Bug reports
- Performance analysis
- Testing specs
- Implementation guides

### Use BOTH in:
- Architecture docs (like this!)
- Tutorials
- Research papers
- Comprehensive guides
- Code docstrings (for key classes)

---

## Code Comment Style

```python
def complex_operation(self, input_data):
    """
    🌊 Transform raw chaos into crystallized understanding.
    ⚙️ Apply 8-stage pipeline: encode → spike → modulate → reason.
    
    Args:
        input_data: Input stimulus
    
    Returns:
        🌊 Enlightened response
        ⚙️ Dict with 'result', 'phase', 'syntropy'
    """
    # 🌊 The first breath—perception awakens
    # ⚙️ Tokenize and embed input text
    encoded = self.encoder.encode(input_data['text'])
    
    # 🌊 Patterns emerge from the noise
    # ⚙️ KMeans clustering on encoded features
    patterns = self.extractor.extract(encoded)
    
    # ... more stages ...
    
    return result
```

---

## Numbers You Should Know

| Value | Poetic Meaning | Mathematical Meaning |
|-------|----------------|---------------------|
| 0.1 | A single tick of the cosmic clock | Phase increment (radians) |
| 0.9 | Memory's gentle persistence | EMA decay constant |
| 1.1 | Growth factor of wisdom | Learning rate multiplier |
| 63 | One cosmic cycle complete | Oscillations for full 2π rotation |
| 7 | Half-remembered past | Memory half-life (steps) |
| 10 | Wisdom's peak | Maximum learning rate multiplier |
| 500 | Depth of symbolic memory | Concept history buffer size |
| 20 | Window of recent awareness | Context window size |

---

## Debugging Translation Guide

**Error:** "Phase value 7.5 out of range"
- 🌊 The cosmic clock has drifted beyond the cycle
- ⚙️ Phase should be [0, 2π), got 7.5 > 2π ≈ 6.28
- Fix: Apply modulo operation `% (2 * np.pi)`

**Error:** "Learning rate 0.15 exceeds maximum"
- 🌊 Wisdom grows too fast—unstable acceleration
- ⚙️ lr exceeded 10x cap (0.01 * 10 = 0.1)
- Fix: Clamp with `min(lr, base_lr * 10)`

**Error:** "Memory file not found"
- 🌊 The eternal archive has vanished
- ⚙️ Disk persistence failed, state not saved
- Fix: Initialize new MemoryStore, creates directory

---

**For full details, see [DUAL_LANGUAGE_GLOSSARY.md](./DUAL_LANGUAGE_GLOSSARY.md)**
