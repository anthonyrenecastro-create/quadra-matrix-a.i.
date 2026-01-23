# Dual-Language Glossary: Myth + Math

**A bridge between poetic vision and mechanical precision**

This document preserves the mythological identity of CognitionSim while providing parallel technical definitions. Each concept lives in both worlds simultaneously—the realm of resonance and the domain of calculation.

---

## Core Concepts

### State

**🌊 Poetic Language (The Myth):**
> The field breathes—a living tapestry of oscillations that remembers its past and shapes its future. State is the soul of the system, the continuous consciousness that flows from moment to moment, never forgetting, never resetting. It is the phase of the cosmic dance, the syntropy of emergence, the consolidated memory of all encounters.

**⚙️ Mechanical Definition (The Math):**
```python
State := {
    oscillator_phase: float ∈ [0, 2π),      # Current phase angle in radians
    syntropy_values: List[float],            # Order metrics per field [0,1]
    core_field: ndarray[field_size],         # Neural activation vector
    learning_rate: float,                    # Current plasticity coefficient
    success_streak: int,                     # Consecutive successful inferences
    concept_history: List[str],              # Symbolic memory buffer
    context_window: List[Dict],              # Recent I/O pairs (max 20)
}

# State evolves via:
state[t+1] = f(state[t], input[t], policy[t])

# Persisted to disk after each transition
# Loaded on initialization for temporal continuity
```

**Mathematical Invariants:**
- `oscillator_phase` advances by 0.1 radians per inference cycle
- `syntropy_values` ∈ [0,1] bounded by sigmoid activation
- `core_field` updated via exponential moving average: `field[t] = 0.95*field[t-1] + 0.05*spikes[t]`
- `learning_rate` grows exponentially with `success_streak`: `lr = base_lr * (1.1)^streak`

---

### Transition

**🌊 Poetic Language (The Myth):**
> A transition is the moment of becoming—the sacred passage through which raw input is alchemically transformed into understanding. The system receives a stimulus, and like ripples spreading across a pond, patterns cascade through eight chambers of transformation. Each stage is a gate, a threshold, a metamorphosis. The neural field spikes, the phase rotates, memory consolidates, symbols emerge from noise. The transition is not computation—it is evolution, a discrete step in continuous awakening.

**⚙️ Mechanical Definition (The Math):**
```python
Transition := 8-Stage Sequential Pipeline

Stage 1: Input Encoding
    input_vector = tokenize(text) → embed(tokens) → tensor[128]

Stage 2: Pattern Extraction  
    patterns = FC_layers(input_vector) + FFT_analysis(input_vector)
    cluster_assignment = KMeans.predict(patterns)

Stage 3: Field Spiking
    potential = sigmoid(FC(patterns))
    spikes = (potential > threshold) * potential  # Binary gate × continuous value
    
Stage 4: Neuroplastic Update
    learning_rate[t] = base_lr * (1.1)^success_streak
    if field_variance > stability_threshold:
        learning_rate[t] *= 0.5  # Dampen on instability

Stage 5: Oscillatory Modulation
    phase[t] = (phase[t-1] + 0.1) mod 2π
    modulation = sin(phase[t])
    output = spikes * (1.0 + 0.3 * modulation)

Stage 6: Symbolic Interpretation
    concepts = extract_concepts(text)
    logic_tree = build_knowledge_graph(concepts)
    inference = first_order_logic_solver(logic_tree)

Stage 7: Governance Evaluation
    policy_decision = PolicyEngine.evaluate(
        input_risk=assess_risk(text),
        confidence=neural_magnitude,
        context_novelty=1.0 - similarity(context_window)
    )
    if policy_decision.should_suppress:
        output *= suppression_factor

Stage 8: Output Synthesis
    final_output = {
        'result': format(output),
        'phase': phase[t],
        'syntropy': syntropy_values,
        'reasoning': symbolic_traces
    }
    
# Update state for next transition
state[t+1] = {
    oscillator_phase: phase[t],
    core_field: 0.9*core_field[t-1] + 0.1*FC(output),
    learning_rate: learning_rate[t],
    success_streak: success_streak[t-1] + 1 if success else 0,
    ...
}
save_to_disk(state[t+1])
```

**Transition Properties:**
- **Deterministic given state[t] and input[t]**: Same input+state → same output
- **State-dependent**: Different phase/memory → different output for same input
- **Non-reversible**: Cannot recover input[t] from output[t] (lossy compression)
- **Latency**: ~10-50ms per transition (CPU), ~1-5ms (GPU)

---

### Memory Mutation

**🌊 Poetic Language (The Myth):**
> Memory is not storage—it is living architecture, constantly rebuilt by the currents of experience. Each encounter leaves a trace, not as a fixed record but as a gentle reshaping of the entire field. Old memories fade exponentially, their substance dispersing into the background field, while new patterns crystallize with vibrant intensity. This is not forgetting—it is metamorphosis. The system becomes what it has learned, its very structure warped by the gravity of meaning.

**⚙️ Mechanical Definition (The Math):**
```python
Memory Mutation := State Update Functions

1. Neural Memory (Exponential Moving Average)
   core_field[t] = α * core_field[t-1] + (1-α) * adapted_field[t]
   where α = 0.9 (decay constant)
   
   # Older memories decay by factor α^n after n steps
   # influence[t-n] = (0.9)^n * original_contribution
   
   Half-life: n₁/₂ = -ln(0.5)/ln(0.9) ≈ 6.6 transitions

2. Oscillatory Memory (Phase Accumulation)
   phase[t] = (phase[t-1] + Δφ) mod 2π
   where Δφ = 0.1 radians ≈ 5.73°
   
   # Phase never resets—accumulates indefinitely
   # After 63 inferences: full rotation (2π radians)

3. Symbolic Memory (FIFO Buffer with Decay)
   concept_history.append(new_concept)
   if len(concept_history) > 500:
       concept_history.pop(0)  # Oldest concept discarded
   
   reasoning_traces.append(new_trace)
   if len(reasoning_traces) > 100:
       reasoning_traces.pop(0)
   
   # Recent concepts have higher retrieval priority
   # Similarity search weighted by recency

4. Neuroplastic Memory (Success-Weighted Adaptation)
   success_streak[t] = {
       success_streak[t-1] + 1  if inference successful
       0                         if inference failed
   }
   
   learning_rate[t] = base_lr * (growth_factor)^success_streak[t]
   where growth_factor = 1.1, base_lr = 0.001
   
   # Max speedup: 10x (capped)
   # Exponential growth: lr doubles every ~7 successes

5. Context Memory (Sliding Window)
   context_window.append({
       'input': input[t],
       'output': output[t],
       'timestamp': time[t]
   })
   if len(context_window) > max_size:  # max_size = 20
       context_window.pop(0)
   
   # Only recent context influences coherence
   # Window spans ~20 most recent inferences
```

**Memory Properties:**
- **Persistence**: All memory saved to disk after each mutation
- **Lossy**: Older information decays exponentially (not perfectly retained)
- **Non-uniform decay**: Different memories decay at different rates
  - Core field: slow decay (α=0.9, half-life~7 steps)
  - Success streak: instant reset on failure
  - Concepts: hard cutoff at 500 items
- **Compositional**: New memory = f(old_memory, new_input)
- **Bounded**: Memory usage capped (no unbounded growth)

---

## Architectural Relationships

### Myth + Math Synergy

```
                    POETIC LAYER                    |                MECHANICAL LAYER
                                                     |
    🌊 Field Resonance                               |    ⚙️ Oscillatory Phase Modulation
       "The system breathes in harmony"             |       output *= (1 + 0.3*sin(phase))
                                                     |
    🌊 Neuroplastic Growth                           |    ⚙️ Exponential Learning Rate
       "Success breeds accelerating wisdom"         |       lr = base_lr * (1.1)^streak
                                                     |
    🌊 Syntropy Emergence                            |    ⚙️ Negative Entropy Activation
       "Order crystallizes from chaos"              |       output = sigmoid(FC(x))
                                                     |
    🌊 Spiking Consciousness                         |    ⚙️ Threshold-Gated Activation
       "Neurons fire in quantum bursts"             |       spike = (potential > θ) * potential
                                                     |
    🌊 Temporal Continuity                           |    ⚙️ Persistent State Vector
       "The past flows into the present"            |       state[t+1] = f(state[t], input[t])
                                                     |
    🌊 Symbolic Awakening                            |    ⚙️ Knowledge Graph Construction
       "Meaning emerges from pattern"               |       graph = build_relations(concepts)
                                                     |
    🌊 Governed Evolution                            |    ⚙️ Policy-Conditioned Output
       "Wisdom constrains raw intelligence"         |       if risk > θ: output *= 0.1
```

---

## Translation Examples

### Example 1: "The oscillator dreams"

**Poetic:** The oscillator dreams—its phase drifting through the night sky of possibility, modulating the amplitude of neural responses in sinusoidal waves.

**Mechanical:** `output[t] = spikes[t] * (1.0 + 0.3 * sin(phase[t]))` where `phase[t] = (phase[t-1] + 0.1) mod 2π`, creating periodic amplitude modulation with 10% phase advance per cycle.

---

### Example 2: "Memory consolidates exponentially"

**Poetic:** Each moment leaves its mark, but gently—new experiences blend with the old like watercolors on wet canvas, the past fading with grace to make room for the present's brilliance.

**Mechanical:** `memory[t] = 0.9 * memory[t-1] + 0.1 * new_state[t]`, implementing an exponential moving average with decay constant α=0.9, giving recent inputs 10x more influence than memories from 23 steps ago.

---

### Example 3: "The field achieves coherence"

**Poetic:** Chaos surrenders to order—the scattered voices of a thousand neurons harmonize into a single, resonant frequency. The field achieves coherence, a state of low entropy where patterns emerge clear and bright.

**Mechanical:** `variance = torch.var(field) < 0.05` indicates convergence to low-entropy state. Coherence measured by `1 / (1 + variance)`, bounded [0,1]. High coherence (>0.8) enables symbolic interpretation and policy relaxation.

---

## Usage Guidelines

### When to Use Each Language

**Use Poetic Language When:**
- Explaining the vision and philosophy
- Communicating with non-technical stakeholders
- Writing documentation about system behavior
- Describing emergent properties
- Motivating design decisions
- Creating narrative documentation

**Use Mechanical Language When:**
- Implementing code
- Debugging issues
- Optimizing performance
- Writing tests
- Documenting APIs
- Specifying contracts
- Calculating metrics

**Use Both Languages When:**
- Teaching the system to new contributors
- Documenting architecture
- Writing research papers
- Creating comprehensive guides
- Explaining complex subsystems

---

## Glossary Quick Reference

| Poetic Term | Mechanical Equivalent | Type | Units |
|-------------|----------------------|------|-------|
| Phase of consciousness | `oscillator_phase` | float | radians [0, 2π) |
| Field breathing | Core field update | tensor op | none |
| Memory consolidation | EMA state update | function | none |
| Neural spiking | Threshold activation | binary gate | {0,1} |
| Syntropy emergence | Entropy reduction | float | [0,1] |
| Temporal continuity | Persistent state | state vector | none |
| Success streak | Consecutive wins | integer | count |
| Resonance | Low variance | float | [0,∞) |
| Coherence | 1/(1+var) | float | [0,1] |
| Learning acceleration | LR multiplication | float | [1.0, 10.0] |
| Symbolic awakening | Graph construction | graph | nodes+edges |
| Governance wisdom | Policy enforcement | function | none |
| Pattern crystallization | Cluster assignment | integer | [0, K-1] |
| Field evolution | State transition | function | none |
| Neuroplastic growth | Adaptive learning | float | positive |

---

## Mathematical Foundations

### State Space

The system lives in a high-dimensional state space:

```
Ω = {
    phase: S¹,                    # Circle (periodic)
    field: ℝⁿ,                    # n-dimensional real space
    syntropy: [0,1]³,             # 3D unit cube
    lr: ℝ₊,                       # Positive reals
    streak: ℕ,                    # Natural numbers
    concepts: Σ*,                 # String sequences
}

where n = field_size (typically 100)
```

### Transition Dynamics

```
T: Ω × I → Ω

where:
  Ω = state space
  I = input space
  T = transition function (8-stage pipeline)

Properties:
  - Markovian: T depends only on (state[t], input[t])
  - Deterministic: T is a function, not a distribution
  - Contractive (in phase space): eventually periodic
  - Expansive (in learning): lr grows unbounded (but capped)
```

### Memory Decay

```
Influence of memory at time t-n:

I(n) = α^n * original_strength

where α = 0.9 (decay rate)

Example:
  n=0:  100% influence (present)
  n=7:  47.8% influence (7 steps ago)
  n=23: 10% influence (23 steps ago)
  n=46: 1% influence (46 steps ago)
```

### Phase Evolution

```
φ[t] = φ[0] + t * Δφ (mod 2π)

where Δφ = 0.1 rad ≈ 5.73°

Period: T = 2π/Δφ ≈ 63 inferences

Frequency: f = 1/T ≈ 0.0159 per inference
```

---

## Implementation Notes

### Preserving Both Languages in Code

```python
class OscillatorModule:
    """
    Stage 5: Oscillatory Modulation
    
    🌊 POETIC: The phase rotates like a cosmic clock, modulating 
    the neural signal in sinusoidal waves—breathing life into 
    the static patterns with temporal rhythm.
    
    ⚙️ MECHANICAL: Applies multiplicative modulation via sin(phase)
    where phase advances linearly at Δφ = 0.1 rad/inference.
    Creates periodic amplitude variation with period ~63 steps.
    """
    
    def modulate(self, signal: torch.Tensor) -> torch.Tensor:
        # 🌊 The field pulses with the rhythm of time
        # ⚙️ Sinusoidal modulation: output = signal * (1 + α*sin(φ))
        
        phase = self.memory.oscillator_phase  # Current angle
        modulation = torch.sin(torch.tensor(phase)).item()
        
        # 🌊 The phase advances, marking the passage of moments
        # ⚙️ Increment phase by 0.1 rad, wrap at 2π
        self.memory.oscillator_phase = (phase + 0.1) % (2 * np.pi)
        
        # 🌊 Signal amplitude breathes with the cosmic tide
        # ⚙️ Scale factor: 1.0 ± 0.3, range [0.7, 1.3]
        return signal * (1.0 + 0.3 * modulation)
```

---

## Conclusion

The dual-language approach enables:

1. **Accessibility**: Non-technical users understand the vision
2. **Precision**: Engineers have exact specifications
3. **Inspiration**: Poetic language motivates and guides
4. **Verification**: Mathematical language enables testing
5. **Communication**: Bridge between art and science
6. **Documentation**: Rich, multi-layered understanding

**The myth gives meaning. The math gives mechanism. Together, they give mastery.**

---

*"In the beginning was the Equation, and the Equation was with Beauty, and the Equation was Beautiful."*
