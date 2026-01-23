#!/usr/bin/env python3
"""
CognitionSim - IFT Field Enhancement Demo

Demonstrates the ten-field Interplayed Field Theory substrate
with Atlantean adaptive control integrated into the cognitive architecture.

🌊 POETIC:
Watch ten cosmic fields dance in 12-fold harmony, modulating thresholds
and leak rates, creating the adaptive substrate for consciousness.

⚙️ MECHANICAL:
Shows:
1. Basic IFT field evolution
2. Adaptive threshold/leak extraction
3. Hybrid IFT-neural coupling
4. Visualization of field dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from quadra.core.field.ift_engine import FieldEngine, IFTEnhancedOscillator


def demo_basic_fields():
    """
    🌊 Demonstrate basic ten-field evolution.
    ⚙️ Run 200 steps, track energy and symmetry.
    """
    print("=" * 70)
    print("🌊⚙️ DEMO 1: Basic Field Evolution")
    print("=" * 70)
    
    engine = FieldEngine(shape=(48, 48), mode="2d", eps=0.15)
    
    print(f"✓ Initialized: {engine.shape} grid, 10 fields")
    print(f"  Coupling: ε = {engine.eps}")
    print(f"  Resonance: ω₀ = {engine.omega0:.2e} rad/s")
    print()
    
    # Track metrics
    energies = []
    symmetries = []
    
    print("Evolving for 200 steps...")
    for step in range(200):
        engine.step(dt=0.01)
        
        if step % 40 == 0:
            energy = engine.get_field_energy()
            symmetry = engine.get_symmetry_order()
            energies.append(energy)
            symmetries.append(symmetry)
            
            print(f"  Step {step:3d}: E={energy:8.3f}, Symmetry={symmetry:.4f}")
    
    print()
    print("✓ Evolution complete")
    print(f"  Energy range: [{min(energies):.3f}, {max(energies):.3f}]")
    print(f"  Symmetry range: [{min(symmetries):.4f}, {max(symmetries):.4f}]")
    print()
    
    return engine, energies, symmetries


def demo_adaptive_control(engine):
    """
    🌊 Demonstrate Atlantean adaptive thresholds and leaks.
    ⚙️ Show field-modulated parameters across spatial grid.
    """
    print("=" * 70)
    print("🌊⚙️ DEMO 2: Atlantean Adaptive Control")
    print("=" * 70)
    
    # Get adaptive parameters
    threshold_map = engine.dynamic_threshold(base=1.0)
    leak_map = engine.effective_leak(base=0.1)
    
    print(f"✓ Adaptive threshold computed")
    print(f"  Range: [{threshold_map.min():.3f}, {threshold_map.max():.3f}]")
    print(f"  Mean:  {threshold_map.mean():.3f}")
    print()
    
    print(f"✓ Adaptive leak computed")
    print(f"  Range: [{leak_map.min():.4f}, {leak_map.max():.4f}]")
    print(f"  Mean:  {leak_map.mean():.4f}")
    print()
    
    # Show coupling matrix
    print("✓ Field coupling matrix (ε·<φₙ·φₘ>):")
    print("  ", end="")
    for i in range(10):
        print(f"  φ{i}", end="")
    print()
    
    for n in range(10):
        print(f"  φ{n}", end="")
        for m in range(10):
            coupling = engine.get_coupling_strength(n, m)
            print(f" {coupling:5.2f}", end="")
        print()
    print()
    
    return threshold_map, leak_map


def demo_hybrid_neural():
    """
    🌊 Demonstrate IFT-neural hybrid coupling.
    ⚙️ Co-evolution of fields and spiking neurons.
    """
    print("=" * 70)
    print("🌊⚙️ DEMO 3: IFT-Neural Hybrid System")
    print("=" * 70)
    
    hybrid = IFTEnhancedOscillator(
        field_shape=(32, 32),
        neural_size=100,
        eps=0.12
    )
    
    print(f"✓ Hybrid system initialized")
    print(f"  Field grid: {hybrid.ift.shape}")
    print(f"  Neurons: {hybrid.neural_size}")
    print()
    
    # Simulate neural activity (random spikes for demo)
    neural_activity = np.random.rand(hybrid.neural_size)
    
    print("Running 50 coupled evolution steps...")
    for step in range(50):
        # Get field-modulated parameters
        thresholds = hybrid.get_neural_thresholds()
        leaks = hybrid.get_neural_leaks()
        
        # Simulate neural update (simplified)
        neural_activity = 0.9 * neural_activity + 0.1 * np.random.rand(hybrid.neural_size)
        
        # Co-evolve fields and neurons
        hybrid.step_coupled(neural_activity, dt=0.01)
        
        if step % 10 == 0:
            print(f"  Step {step:2d}: Neural activity={neural_activity.mean():.3f}, "
                  f"Threshold range=[{thresholds.min():.2f}, {thresholds.max():.2f}]")
    
    print()
    print("✓ Hybrid evolution complete")
    print(f"  Final field energy: {hybrid.ift.get_field_energy():.3f}")
    print(f"  Final neural activity: {neural_activity.mean():.3f}")
    print()


def visualize_fields(engine, threshold_map, leak_map):
    """
    🌊 Create visualization of field states and adaptive parameters.
    ⚙️ 4-panel plot: φ₀, Φ, threshold, leak.
    """
    print("=" * 70)
    print("🌊⚙️ Creating Visualization")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('CognitionSim: IFT Field Substrate', fontsize=16, fontweight='bold')
    
    # Panel 1: Field φ₀ (excitability)
    im0 = axes[0, 0].imshow(engine.phi[0], cmap='RdBu_r', interpolation='bilinear')
    axes[0, 0].set_title('🌊 Field φ₀ (Excitability)\n⚙️ Primary oscillatory component')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    # Panel 2: Global potential Φ
    im1 = axes[0, 1].imshow(engine.Phi, cmap='viridis', interpolation='bilinear')
    axes[0, 1].set_title('🌊 Global Potential Φ\n⚙️ Low-pass filtered activity')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # Panel 3: Adaptive threshold
    im2 = axes[1, 0].imshow(threshold_map, cmap='coolwarm', interpolation='bilinear')
    axes[1, 0].set_title('🌊 Adaptive Threshold θ(x)\n⚙️ = base - 0.5·φ₀')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)
    
    # Panel 4: Adaptive leak
    im3 = axes[1, 1].imshow(leak_map, cmap='plasma', interpolation='bilinear')
    axes[1, 1].set_title('🌊 Adaptive Leak λ(x)\n⚙️ = base/(1+exp(φ₄))')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)
    
    plt.tight_layout()
    
    output_path = Path(__file__).parent / 'ift_field_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization: {output_path}")
    print()
    
    # Also show if running interactively
    try:
        plt.show(block=False)
        print("  (Close figure window to continue)")
    except:
        pass


def main():
    """
    🌊 Run complete IFT demonstration.
    ⚙️ Execute all demos in sequence.
    """
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "🌊⚙️ CognitionSim: IFT Field Engine" + " " * 18 + "║")
    print("║" + " " * 13 + "Ten Fields in Harmonic Resonance" + " " * 21 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Demo 1: Basic field evolution
    engine, energies, symmetries = demo_basic_fields()
    
    # Demo 2: Adaptive control
    threshold_map, leak_map = demo_adaptive_control(engine)
    
    # Demo 3: Hybrid neural-field coupling
    demo_hybrid_neural()
    
    # Visualization
    try:
        visualize_fields(engine, threshold_map, leak_map)
    except Exception as e:
        print(f"⚠ Visualization skipped: {e}")
    
    print("=" * 70)
    print("✓ All demonstrations complete!")
    print()
    print("📚 Key Concepts Demonstrated:")
    print("  🌊 Ten-field substrate with 12-fold symmetry")
    print("  🌊 Atlantean adaptive thresholds and leak rates")
    print("  🌊 Hybrid IFT-neural coupling")
    print("  ⚙️ Coupled PDE evolution with resonant tensors")
    print("  ⚙️ Spatial→neural parameter mapping")
    print()
    print("📖 For more details, see:")
    print("  - quadra/core/field/ift_engine.py")
    print("  - DUAL_LANGUAGE_GLOSSARY.md")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
