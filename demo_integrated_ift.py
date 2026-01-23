"""
Demo: Integrated IFT Field Substrate in CognitionSim

Demonstrates the complete integration of the ten-field IFT substrate
with the stateful symbolic predictive interpreter.

Shows:
1. Basic inference with IFT enabled/disabled
2. Field evolution over multiple inferences
3. Adaptive parameter modulation (threshold, leak)
4. Field metrics tracking (energy, symmetry, coupling)
5. Memory persistence across sessions
"""

import asyncio
import numpy as np
import matplotlib.pyplot as plt
from quadra.core.symbolic.interpreter import StatefulSymbolicPredictiveInterpreter


async def demo_basic_inference():
    """Demo 1: Basic inference with IFT enabled vs disabled."""
    print("=" * 70)
    print("DEMO 1: Basic Inference - IFT Enabled vs Disabled")
    print("=" * 70)
    
    # Create interpreters
    interpreter_ift = StatefulSymbolicPredictiveInterpreter(
        model_version="demo-ift-1.0",
        enable_ift=True,
        field_shape=(16, 16)  # Small field for demo
    )
    
    interpreter_baseline = StatefulSymbolicPredictiveInterpreter(
        model_version="demo-baseline-1.0",
        enable_ift=False
    )
    
    # Test input
    test_input = {
        'text': "consciousness emerges from field dynamics",
        'concepts': ['consciousness', 'emergence', 'field']
    }
    
    # Run inference
    print("\n🌊 Running with IFT substrate...\n")
    result_ift = await interpreter_ift.process(test_input, request_id="ift-demo-1")
    
    print("\n⚙️ Running without IFT substrate...\n")
    result_baseline = await interpreter_baseline.process(test_input, request_id="baseline-demo-1")
    
    # Compare results
    print("\n" + "─" * 70)
    print("COMPARISON: IFT vs Baseline")
    print("─" * 70)
    
    print(f"\n🌊 WITH IFT:")
    print(f"  • Neural magnitude: {result_ift['neural_magnitude']:.4f}")
    print(f"  • Spike rate: {result_ift['spike_rate']:.4f}")
    print(f"  • Oscillatory phase: {result_ift['oscillatory_phase']:.4f}")
    
    if 'ift_field_metrics' in result_ift:
        ift_metrics = result_ift['ift_field_metrics']
        print(f"  • Field energy: {ift_metrics['field_energy']:.4f}")
        print(f"  • Symmetry order: {ift_metrics['symmetry_order']:.4f}")
        print(f"  • Global potential: {ift_metrics['global_potential']:.4f}")
    
    if 'adaptive_parameters' in result_ift:
        params = result_ift['adaptive_parameters']
        print(f"  • Adaptive threshold: {params['threshold']:.4f}")
        print(f"  • Adaptive leak: {params['leak']:.4f}")
    
    print(f"\n⚙️ WITHOUT IFT:")
    print(f"  • Neural magnitude: {result_baseline['neural_magnitude']:.4f}")
    print(f"  • Spike rate: {result_baseline['spike_rate']:.4f}")
    print(f"  • Oscillatory phase: {result_baseline['oscillatory_phase']:.4f}")
    
    print("\n" + "─" * 70)
    print("Symbolic results:")
    print("─" * 70)
    print(f"IFT:      {result_ift['symbolic_result'][:100]}...")
    print(f"Baseline: {result_baseline['symbolic_result'][:100]}...")
    
    return interpreter_ift, result_ift


async def demo_field_evolution():
    """Demo 2: Track field evolution over multiple inferences."""
    print("\n\n" + "=" * 70)
    print("DEMO 2: Field Evolution Over Multiple Inferences")
    print("=" * 70)
    
    interpreter = StatefulSymbolicPredictiveInterpreter(
        model_version="demo-evolution-1.0",
        enable_ift=True,
        field_shape=(32, 32)
    )
    
    # Track metrics over time
    energies = []
    symmetries = []
    potentials = []
    thresholds = []
    leaks = []
    phases = []
    
    test_inputs = [
        {'text': 'pattern A', 'concepts': ['pattern']},
        {'text': 'pattern B emerges', 'concepts': ['pattern', 'emergence']},
        {'text': 'coherence across fields', 'concepts': ['coherence', 'field']},
        {'text': 'resonance builds', 'concepts': ['resonance']},
        {'text': 'memory crystallizes', 'concepts': ['memory', 'crystallization']},
    ]
    
    print("\nRunning sequence of inferences...\n")
    
    for i, test_input in enumerate(test_inputs):
        result = await interpreter.process(test_input, request_id=f"evolution-{i}")
        
        # Extract metrics
        if 'ift_field_metrics' in result:
            ift = result['ift_field_metrics']
            energies.append(ift['field_energy'])
            symmetries.append(ift['symmetry_order'])
            potentials.append(ift['global_potential'])
        
        if 'adaptive_parameters' in result:
            params = result['adaptive_parameters']
            thresholds.append(params['threshold'])
            leaks.append(params['leak'])
        
        phases.append(result['oscillatory_phase'])
        
        print(f"Step {i+1}: {test_input['concepts']}")
        print(f"  Energy: {energies[-1]:.4f}, Symmetry: {symmetries[-1]:.4f}, "
              f"Threshold: {thresholds[-1]:.4f}, Leak: {leaks[-1]:.4f}")
    
    # Plot evolution
    print("\nGenerating evolution plot...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    steps = range(1, len(energies) + 1)
    
    axes[0, 0].plot(steps, energies, 'o-', color='#e74c3c')
    axes[0, 0].set_title('Field Energy')
    axes[0, 0].set_xlabel('Inference Step')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(steps, symmetries, 'o-', color='#3498db')
    axes[0, 1].set_title('12-fold Symmetry Order')
    axes[0, 1].set_xlabel('Inference Step')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(steps, potentials, 'o-', color='#9b59b6')
    axes[0, 2].set_title('Global Potential Φ')
    axes[0, 2].set_xlabel('Inference Step')
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].plot(steps, thresholds, 'o-', color='#e67e22')
    axes[1, 0].set_title('Adaptive Threshold θ(φ₀)')
    axes[1, 0].set_xlabel('Inference Step')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(steps, leaks, 'o-', color='#1abc9c')
    axes[1, 1].set_title('Adaptive Leak λ(φ₄)')
    axes[1, 1].set_xlabel('Inference Step')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].plot(steps, phases, 'o-', color='#34495e')
    axes[1, 2].set_title('Oscillatory Phase φ')
    axes[1, 2].set_xlabel('Inference Step')
    axes[1, 2].set_ylim([0, 2*np.pi])
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle('IFT Field Evolution Across Inferences', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ift_evolution.png', dpi=150, bbox_inches='tight')
    print("✓ Saved plot: ift_evolution.png")
    
    return interpreter, energies, symmetries


async def demo_adaptive_control():
    """Demo 3: Show how fields modulate neural parameters."""
    print("\n\n" + "=" * 70)
    print("DEMO 3: Adaptive Control - Field-Modulated Parameters")
    print("=" * 70)
    
    interpreter = StatefulSymbolicPredictiveInterpreter(
        model_version="demo-adaptive-1.0",
        enable_ift=True,
        field_shape=(32, 32)
    )
    
    # Run several inferences to build up field dynamics
    print("\nBuilding field dynamics (20 inferences)...\n")
    
    for i in range(20):
        await interpreter.process(
            {'text': f'evolving pattern {i}', 'concepts': ['pattern', 'evolution']},
            request_id=f"warmup-{i}"
        )
    
    # Now check the adaptive parameters
    print("Field substrate has evolved. Checking adaptive parameters...\n")
    
    # Access field engine directly
    field_engine = interpreter.oscillator.field_engine
    
    if field_engine is not None:
        # Sample spatial variation in threshold and leak
        H, W = field_engine.shape
        
        # Get full spatial maps
        threshold_map = field_engine.dynamic_threshold(base=1.0)
        leak_map = field_engine.effective_leak(base=0.1)
        
        print(f"Threshold map (φ₀-modulated):")
        print(f"  Shape: {threshold_map.shape}")
        print(f"  Range: [{threshold_map.min():.4f}, {threshold_map.max():.4f}]")
        print(f"  Mean: {threshold_map.mean():.4f} (base=1.0)")
        print(f"  Std: {threshold_map.std():.4f}")
        
        print(f"\nLeak map (φ₄-modulated):")
        print(f"  Shape: {leak_map.shape}")
        print(f"  Range: [{leak_map.min():.4f}, {leak_map.max():.4f}]")
        print(f"  Mean: {leak_map.mean():.4f} (base=0.1)")
        print(f"  Std: {leak_map.std():.4f}")
        
        # Visualize spatial maps
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # φ₀ (excitability)
        im0 = axes[0].imshow(field_engine.phi[0], cmap='RdBu_r', aspect='auto')
        axes[0].set_title('φ₀ (Excitability Field)')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046)
        
        # Threshold map
        im1 = axes[1].imshow(threshold_map, cmap='viridis', aspect='auto')
        axes[1].set_title('Adaptive Threshold: θ = 1.0 - 0.5·φ₀')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        
        # Leak map
        im2 = axes[2].imshow(leak_map, cmap='plasma', aspect='auto')
        axes[2].set_title('Adaptive Leak: λ = 0.1/(1 + exp(φ₄))')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)
        
        plt.suptitle('Atlantean Adaptive Control Maps', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('ift_adaptive_control.png', dpi=150, bbox_inches='tight')
        print("\n✓ Saved plot: ift_adaptive_control.png")
        
        # Show how these would affect a hypothetical neuron at different locations
        print("\n" + "─" * 70)
        print("Spatial Heterogeneity Example:")
        print("─" * 70)
        print(f"\nNeuron at position (5, 5):")
        print(f"  Threshold: {threshold_map[5, 5]:.4f}")
        print(f"  Leak rate: {leak_map[5, 5]:.4f}")
        
        print(f"\nNeuron at position ({H//2}, {W//2}) [center]:")
        print(f"  Threshold: {threshold_map[H//2, W//2]:.4f}")
        print(f"  Leak rate: {leak_map[H//2, W//2]:.4f}")
        
        print(f"\nNeuron at position ({H-5}, {W-5}):")
        print(f"  Threshold: {threshold_map[H-5, W-5]:.4f}")
        print(f"  Leak rate: {leak_map[H-5, W-5]:.4f}")
        
        print("\n🌊 Different locations → different dynamics (spatial heterogeneity)")
        print("⚙️ Same neuron model, adaptive parameters from field substrate")


async def demo_memory_persistence():
    """Demo 4: Show memory persistence across sessions."""
    print("\n\n" + "=" * 70)
    print("DEMO 4: Memory Persistence - Field State Across Sessions")
    print("=" * 70)
    
    # Session 1: Build up state
    print("\nSession 1: Building up field state...")
    interpreter1 = StatefulSymbolicPredictiveInterpreter(
        model_version="demo-persistence-1.0",
        enable_ift=True,
        field_shape=(16, 16)
    )
    
    for i in range(5):
        await interpreter1.process(
            {'text': f'building memory {i}', 'concepts': ['memory', 'build']},
            request_id=f"session1-{i}"
        )
    
    snapshot1 = interpreter1.get_memory_snapshot()
    print(f"  Final phase: {snapshot1['oscillator_phase']:.4f}")
    print(f"  Total concepts: {snapshot1['concept_count']}")
    print(f"  Success streak: {snapshot1['neuroplastic_metrics']['success_streak']}")
    
    if 'ift_field_energy' in interpreter1.memory.get_ift_metrics():
        print(f"  Field energy: {interpreter1.memory.ift_field_energy:.4f}")
    
    # Session 2: Load from same memory store
    print("\nSession 2: Loading from persisted memory...")
    interpreter2 = StatefulSymbolicPredictiveInterpreter(
        model_version="demo-persistence-1.0",
        enable_ift=True,
        field_shape=(16, 16)
    )
    
    snapshot2 = interpreter2.get_memory_snapshot()
    print(f"  Loaded phase: {snapshot2['oscillator_phase']:.4f}")
    print(f"  Loaded concepts: {snapshot2['concept_count']}")
    print(f"  Loaded streak: {snapshot2['neuroplastic_metrics']['success_streak']}")
    
    # Verify continuity
    print("\n" + "─" * 70)
    print("Persistence Check:")
    print("─" * 70)
    phase_match = abs(snapshot1['oscillator_phase'] - snapshot2['oscillator_phase']) < 0.01
    concept_match = snapshot1['concept_count'] == snapshot2['concept_count']
    
    print(f"  Phase preserved: {'✓' if phase_match else '✗'}")
    print(f"  Concepts preserved: {'✓' if concept_match else '✗'}")
    print(f"  Recent concepts: {snapshot2['recent_concepts'][-3:]}")
    
    print("\n🌊 Memory flows across the boundaries of sessions")
    print("⚙️ Disk-backed state enables true temporal continuity")


async def main():
    """Run all demos."""
    print("\n" + "═" * 70)
    print("  IFT Field Substrate Integration Demo")
    print("  CognitionSim - Stateful Symbolic Predictive Interpreter")
    print("═" * 70)
    
    try:
        # Demo 1: Basic comparison
        interpreter, result = await demo_basic_inference()
        
        # Demo 2: Evolution tracking
        interpreter_evo, energies, symmetries = await demo_field_evolution()
        
        # Demo 3: Adaptive control
        await demo_adaptive_control()
        
        # Demo 4: Persistence
        await demo_memory_persistence()
        
        print("\n\n" + "═" * 70)
        print("  All Demos Complete!")
        print("═" * 70)
        print("\n📊 Generated plots:")
        print("  • ift_evolution.png - Field metrics over time")
        print("  • ift_adaptive_control.png - Spatial parameter maps")
        
        print("\n🌊⚙️ IFT substrate fully integrated into CognitionSim!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
