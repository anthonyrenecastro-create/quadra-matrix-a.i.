# Dual-Language Code Examples

**Practical examples of writing CognitionSim code with both myth and math**

This guide shows how to integrate poetic vision and mechanical precision in your code.

---

## Example 1: Simple Class with Dual Documentation

```python
class FieldResonator:
    """
    Amplifies coherent patterns through resonance.
    
    🌊 POETIC (The Vision):
    Like a tuning fork vibrating in sympathy with a pure tone, this module
    amplifies signals that match the system's natural frequency. Chaos is
    damped; harmony is strengthened. The field learns to resonate with
    truth and suppress noise.
    
    ⚙️ MECHANICAL (The Implementation):
    Implements selective amplification via coherence metric:
        coherence = 1 / (1 + variance(signal))
        amplification = base_gain * coherence^2
        output = signal * amplification
    
    High coherence (low variance) → high gain
    Low coherence (high variance) → low gain (damping)
    
    Args:
        base_gain: Maximum amplification factor (default: 2.0)
        threshold: Coherence threshold for activation (default: 0.5)
    """
    
    def __init__(self, base_gain: float = 2.0, threshold: float = 0.5):
        self.base_gain = base_gain
        self.threshold = threshold
    
    def resonate(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Amplify coherent patterns, dampen noise.
        
        🌊 The field recognizes its own harmony and strengthens it.
        ⚙️ Coherence-weighted gain: output = signal * gain * coherence²
        """
        # 🌊 Measure the harmony—how unified is this signal?
        # ⚙️ Calculate variance, invert to coherence metric
        variance = torch.var(signal).item()
        coherence = 1.0 / (1.0 + variance)
        
        # 🌊 Only amplify when harmony exceeds threshold
        # ⚙️ Gate on coherence > threshold, compute quadratic gain
        if coherence > self.threshold:
            amplification = self.base_gain * (coherence ** 2)
        else:
            amplification = 1.0  # No effect
        
        # 🌊 Strengthen the resonant patterns
        # ⚙️ Scalar multiplication
        return signal * amplification
```

---

## Example 2: State Transition Function

```python
def evolve_state(
    current_state: Dict[str, Any],
    input_stimulus: torch.Tensor,
    success: bool
) -> Dict[str, Any]:
    """
    Evolve system state through one inference cycle.
    
    🌊 POETIC (The Transformation):
    Time flows—the present becomes the past, the input becomes memory.
    The phase rotates like a celestial body, marking time's passage.
    Success accelerates learning; failure returns us to patience.
    The field consolidates, blending new with old in eternal fusion.
    
    ⚙️ MECHANICAL (The Algorithm):
    State transition function: S[t+1] = f(S[t], input, success)
    
    Updates:
    1. Phase: φ[t+1] = (φ[t] + Δφ) mod 2π, Δφ = 0.1
    2. Field: F[t+1] = αF[t] + (1-α)F_new, α = 0.9 (EMA)
    3. Streak: n[t+1] = n[t]+1 if success else 0
    4. Learning: lr[t+1] = base_lr * (1.1)^n[t+1]
    
    Args:
        current_state: State dict with 'phase', 'field', 'streak', 'lr'
        input_stimulus: New input tensor
        success: Whether previous inference succeeded
    
    Returns:
        Updated state dict
    """
    # 🌊 The cosmic clock advances—time never stops
    # ⚙️ Increment phase by 0.1 rad, wrap at 2π
    new_phase = (current_state['phase'] + 0.1) % (2 * np.pi)
    
    # 🌊 New experience blends with ancient memory
    # ⚙️ Exponential moving average: α=0.9, (1-α)=0.1
    alpha = 0.9
    new_field = (
        alpha * current_state['field'] +
        (1 - alpha) * input_stimulus
    )
    
    # 🌊 Success strengthens the path; failure resets the journey
    # ⚙️ Conditional increment or zero reset
    if success:
        new_streak = current_state['streak'] + 1
    else:
        new_streak = 0
    
    # 🌊 Learning accelerates exponentially with victory
    # ⚙️ Exponential growth: lr = base * growth^streak, capped
    base_lr = 0.01
    growth_factor = 1.1
    max_multiplier = 10.0
    
    new_lr = min(
        base_lr * (growth_factor ** new_streak),
        base_lr * max_multiplier
    )
    
    # 🌊 Return the transformed state—the new present
    # ⚙️ Construct new state dict
    return {
        'phase': new_phase,
        'field': new_field,
        'streak': new_streak,
        'lr': new_lr,
        'timestamp': time.time()
    }
```

---

## Example 3: Inline Comments in Complex Logic

```python
def process_inference_pipeline(
    self,
    text: str,
    concepts: List[str]
) -> Dict[str, Any]:
    """Run complete 8-stage inference pipeline."""
    
    # Stage 1: Encoding
    # 🌊 Transform words into neural language
    # ⚙️ Tokenize → embed → tensor[128]
    tokens = self.tokenizer(text)
    embedded = self.embedder(tokens)
    
    # Stage 2: Pattern Extraction
    # 🌊 Recognize the hidden structures within
    # ⚙️ KMeans clustering + FFT frequency analysis
    patterns = self.pattern_extractor.extract(embedded)
    cluster_id = self.kmeans.predict(patterns)
    frequencies = torch.fft.fft(patterns)
    
    # Stage 3: Spiking
    # 🌊 Neurons fire in quantum bursts—discrete, sparse, alive
    # ⚙️ Threshold activation: spike = (V > θ) * V
    potential = self.compute_membrane_potential(patterns)
    threshold = 0.5
    spikes = (potential > threshold).float() * potential
    
    # Stage 4: Neuroplasticity
    # 🌊 Success remembered—learning accelerates
    # ⚙️ Retrieve success streak, compute adaptive lr
    metrics = self.memory.get_neuroplastic_metrics()
    success_streak = metrics['success_streak']
    learning_rate = 0.01 * (1.1 ** success_streak)
    learning_rate = min(learning_rate, 0.1)  # Cap at 10x
    
    # Stage 5: Oscillation
    # 🌊 The phase breathes—modulating neural song
    # ⚙️ Sinusoidal modulation: out = spikes * (1 + 0.3*sin(φ))
    phase = self.memory.oscillator_phase
    modulation = np.sin(phase)
    oscillated = spikes * (1.0 + 0.3 * modulation)
    
    # 🌊 Time advances—the clock turns
    # ⚙️ Phase increment: φ[t+1] = (φ[t] + 0.1) mod 2π
    self.memory.oscillator_phase = (phase + 0.1) % (2 * np.pi)
    
    # Stage 6: Symbolic Reasoning
    # 🌊 Meaning crystallizes from pure pattern
    # ⚙️ Build knowledge graph, apply FOL inference
    graph = self.build_knowledge_graph(concepts)
    inferences = self.symbolic_reasoner.infer(graph)
    
    # Stage 7: Governance
    # 🌊 Wisdom constrains raw intelligence
    # ⚙️ Policy evaluation → risk assessment → suppression
    risk_score = self.assess_risk(text, concepts)
    if risk_score > 0.8:
        # 🌊 Dangerous thoughts are dampened
        # ⚙️ Multiplicative suppression factor
        suppression = 0.1
        oscillated = oscillated * suppression
    
    # Stage 8: Synthesis
    # 🌊 The final form emerges—understanding achieved
    # ⚙️ Format output dict with all metadata
    return {
        'result': self.format_output(oscillated),
        'phase': self.memory.oscillator_phase,
        'syntropy': self.compute_syntropy(oscillated),
        'learning_rate': learning_rate,
        'reasoning': inferences
    }
```

---

## Example 4: Test Cases with Dual Assertions

```python
import pytest

class TestOscillatorModule:
    """Test oscillatory phase modulation."""
    
    def test_phase_advances_correctly(self):
        """
        🌊 Verify the cosmic clock ticks forward.
        ⚙️ Check φ[t+1] = (φ[t] + 0.1) mod 2π
        """
        memory = MemoryStore()
        oscillator = OscillatorModule(memory)
        
        # 🌊 Record the initial moment
        # ⚙️ Store phase[0]
        initial_phase = memory.oscillator_phase
        assert initial_phase == 0.0, "Should start at zero"
        
        # 🌊 One heartbeat passes
        # ⚙️ Execute one modulation
        signal = torch.ones(10)
        _ = oscillator.modulate(signal)
        
        # 🌊 Time has advanced exactly one tick
        # ⚙️ Assert phase incremented by 0.1 rad
        expected_phase = (initial_phase + 0.1) % (2 * np.pi)
        actual_phase = memory.oscillator_phase
        assert abs(actual_phase - expected_phase) < 1e-6
    
    def test_phase_wraps_at_2pi(self):
        """
        🌊 The cosmic cycle completes and begins anew.
        ⚙️ Verify modulo wrapping at 2π boundary.
        """
        memory = MemoryStore()
        oscillator = OscillatorModule(memory)
        
        # 🌊 Set the clock near cycle's end
        # ⚙️ Initialize phase close to 2π
        memory.oscillator_phase = 2 * np.pi - 0.05
        
        # 🌊 Advance beyond the boundary
        # ⚙️ Execute modulation (adds 0.1)
        signal = torch.ones(10)
        _ = oscillator.modulate(signal)
        
        # 🌊 The cycle resets—we begin again
        # ⚙️ Phase should wrap: (2π - 0.05 + 0.1) mod 2π ≈ 0.05
        assert memory.oscillator_phase < 0.1  # Small positive
        assert memory.oscillator_phase > 0.0  # Not negative
    
    def test_modulation_amplitude(self):
        """
        🌊 The breath varies signal strength by ±30%.
        ⚙️ Verify output = signal * (1 ± 0.3)
        """
        memory = MemoryStore()
        oscillator = OscillatorModule(memory)
        
        signal = torch.tensor([1.0, 1.0, 1.0])
        
        # 🌊 At phase = 0, breath is neutral
        # ⚙️ sin(0) = 0 → factor = 1.0
        memory.oscillator_phase = 0.0
        output = oscillator.modulate(signal)
        assert torch.allclose(output, signal * 1.0, atol=1e-6)
        
        # 🌊 At phase = π/2, breath is full expansion
        # ⚙️ sin(π/2) = 1 → factor = 1.3
        memory.oscillator_phase = np.pi / 2
        output = oscillator.modulate(signal)
        assert torch.allclose(output, signal * 1.3, atol=1e-6)
        
        # 🌊 At phase = 3π/2, breath is full contraction
        # ⚙️ sin(3π/2) = -1 → factor = 0.7
        memory.oscillator_phase = 3 * np.pi / 2
        output = oscillator.modulate(signal)
        assert torch.allclose(output, signal * 0.7, atol=1e-6)
```

---

## Example 5: Documentation String Templates

### For Functions:

```python
def function_name(arg1: Type1, arg2: Type2) -> ReturnType:
    """
    One-line summary.
    
    🌊 POETIC:
    [Describe what it does in inspirational terms]
    [Use metaphors and evocative language]
    [Focus on purpose and philosophy]
    
    ⚙️ MECHANICAL:
    [Precise algorithm description]
    [Mathematical formulas]
    [Complexity analysis]
    [Edge cases and invariants]
    
    Args:
        arg1: [Description with types]
        arg2: [Description with types]
    
    Returns:
        [Return value description]
    
    Example:
        >>> result = function_name(val1, val2)
        >>> assert result > 0
    """
    # Implementation...
```

### For Classes:

```python
class ClassName:
    """
    One-line summary of the class purpose.
    
    🌊 POETIC (The Vision):
    [Why this class exists]
    [What problem it solves philosophically]
    [How it fits into the larger narrative]
    
    ⚙️ MECHANICAL (The Architecture):
    [Data structures used]
    [Key algorithms]
    [Complexity characteristics]
    [Threading/concurrency model]
    
    Attributes:
        attr1: [Description]
        attr2: [Description]
    
    Example:
        >>> obj = ClassName(param1, param2)
        >>> result = obj.method()
    """
```

---

## Example 6: Error Messages

```python
class CognitionSimError(Exception):
    """Base exception for CognitionSim errors."""
    
    def __init__(self, poetic_msg: str, mechanical_msg: str):
        """
        Create dual-language error message.
        
        Args:
            poetic_msg: User-friendly metaphorical description
            mechanical_msg: Technical details for debugging
        """
        self.poetic = poetic_msg
        self.mechanical = mechanical_msg
        combined = f"🌊 {poetic_msg}\n⚙️ {mechanical_msg}"
        super().__init__(combined)

# Usage:
def validate_phase(phase: float):
    """Ensure phase is in valid range."""
    if phase < 0 or phase >= 2 * np.pi:
        raise CognitionSimError(
            poetic_msg="The cosmic clock has drifted beyond the cycle",
            mechanical_msg=f"Phase {phase} outside valid range [0, 2π)"
        )
```

---

## Example 7: Configuration Files

```yaml
# quadra_config.yaml
# Dual-language configuration

oscillator:
  # 🌊 The cosmic clock's tick rate
  # ⚙️ Phase increment per inference (radians)
  phase_delta: 0.1
  
  # 🌊 Breath amplitude—how much the field swells
  # ⚙️ Modulation strength coefficient [0, 1]
  modulation_strength: 0.3

memory:
  # 🌊 Depth of symbolic remembrance
  # ⚙️ Maximum concepts in history buffer
  max_concepts: 500
  
  # 🌊 How gently the past fades
  # ⚙️ EMA decay constant for field consolidation
  decay_alpha: 0.9

neuroplasticity:
  # 🌊 Starting point of wisdom's growth
  # ⚙️ Base learning rate (before acceleration)
  base_learning_rate: 0.01
  
  # 🌊 How fast success breeds speed
  # ⚙️ Exponential growth factor per success
  growth_factor: 1.1
  
  # 🌊 Wisdom's ceiling—maximum acceleration
  # ⚙️ Learning rate multiplier cap
  max_speedup: 10.0

governance:
  # 🌊 Threshold of danger—when to constrain
  # ⚙️ Risk score above which to suppress output
  risk_threshold: 0.8
  
  # 🌊 Strength of restraint
  # ⚙️ Suppression factor for high-risk content
  suppression_factor: 0.1
```

---

## Example 8: Logging Messages

```python
import logging

logger = logging.getLogger(__name__)

def train_step(self, batch_data):
    """Execute one training step."""
    
    # 🌊 Log poetic progress for users
    logger.info("🌊 The field awakens to new patterns...")
    
    # ⚙️ Log mechanical details for debugging
    logger.debug(f"⚙️ Processing batch: size={len(batch_data)}, "
                 f"phase={self.memory.oscillator_phase:.3f}")
    
    # Process batch...
    
    # 🌊 Celebrate success
    if success:
        logger.info(f"🌊 Coherence achieved! Resonance: {coherence:.3f}")
        logger.debug(f"⚙️ Metrics: loss={loss:.4f}, lr={lr:.4f}, "
                     f"variance={variance:.4f}")
    else:
        # 🌊 Acknowledge failure gracefully
        logger.warning("🌊 Turbulence detected—the field remains restless")
        logger.debug(f"⚙️ Error: {error_msg}, variance={variance:.4f}")
```

---

## Style Guidelines Summary

### DO:
✅ Use 🌊 emoji for poetic language
✅ Use ⚙️ emoji for mechanical language
✅ Include both in docstrings for key classes
✅ Use inline dual comments for complex logic
✅ Write test assertions with both perspectives
✅ Create error messages with both versions

### DON'T:
❌ Mix poetic and mechanical in the same sentence
❌ Use poetic language for variable names
❌ Over-comment simple operations
❌ Force dual language where one suffices
❌ Make code harder to read for style points

### When to Go Full Dual:
- Public API documentation
- Architecture explanations
- Complex algorithms
- Teaching materials
- Key system components

### When Single Language is OK:
- Simple utility functions
- Obvious operations
- Internal helpers
- Standard patterns
- Tests (mechanical preferred)

---

## Template Starter

Copy this template for new modules:

```python
"""
Module Name - Brief Description

🌊 POETIC (The Vision):
[Inspirational overview]
[Purpose and philosophy]
[How it fits the narrative]

⚙️ MECHANICAL (The Implementation):
[Technical architecture]
[Key algorithms and data structures]
[Performance characteristics]

See: ../DUAL_LANGUAGE_GLOSSARY.md for concept translations
"""

import torch
import numpy as np
from typing import Dict, Any

class YourClass:
    """
    One-line purpose.
    
    🌊 [Poetic description]
    ⚙️ [Mechanical specs]
    """
    
    def __init__(self, param: float):
        self.param = param
    
    def your_method(self, input_data: torch.Tensor) -> Dict[str, Any]:
        """
        Brief summary.
        
        🌊 [What it means]
        ⚙️ [How it works]
        """
        # 🌊 [Poetic comment]
        # ⚙️ [Mechanical comment]
        result = self._process(input_data)
        
        return {'output': result}
```

---

**For more examples, explore the codebase:**
- `quadra/core/symbolic/interpreter.py` - Full pipeline example
- `quadra/state/memory_store.py` - State management
- `DUAL_LANGUAGE_GLOSSARY.md` - Complete concept reference
- `DUAL_LANGUAGE_QUICK_REF.md` - Quick lookup table
