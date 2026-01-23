# IFT Quick Reference

## Usage

```python
# Basic usage
from quadra.core.symbolic.interpreter import StatefulSymbolicPredictiveInterpreter

interpreter = StatefulSymbolicPredictiveInterpreter(
    enable_ift=True,
    field_shape=(32, 32)
)

result = await interpreter.process({
    'text': 'your input text',
    'concepts': ['concept1', 'concept2']
})

# Access field metrics
print(result['ift_field_metrics'])
print(result['adaptive_parameters'])
```

## Field Mappings

| Field | Controls | Formula |
|-------|----------|---------|
| φ₀ | Threshold | θ = base - 0.5·φ₀ |
| φ₄ | Leak | λ = base/(1+exp(φ₄)) |
| φ₆ | Learning | η = base·f(φ₆) |

## Commands

```bash
# Test integration
python test_ift_integration.py

# Run full demo
python demo_integrated_ift.py

# Field-only demo
python demo_ift_fields.py
```

## Files

- **Implementation:** [quadra/core/field/ift_engine.py](quadra/core/field/ift_engine.py)
- **Integration:** [quadra/core/symbolic/interpreter.py](quadra/core/symbolic/interpreter.py)
- **Memory:** [quadra/state/memory_store.py](quadra/state/memory_store.py)
- **Docs:** [IFT_INTEGRATION.md](IFT_INTEGRATION.md)

## Key Features

- 10 coupled differential equations with cosine coupling
- Discrete Laplacian and bi-Laplacian operators
- Persistent memory via pickle serialization
- Parameter modulation from field averages
- Spatial variation on 2D grid
- Integration tests pass without errors
