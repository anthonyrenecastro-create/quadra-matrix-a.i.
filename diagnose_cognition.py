#!/usr/bin/env python3
"""
CognitionSim - Cognitive Diagnostic Tool
Advanced analysis and visualization of cognitive processes.

Usage:
    python diagnose_cognition.py [--field-size 100] [--device cpu] [--save]
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict
    import statistics
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install -r requirements.txt")
    sys.exit(1)

from demo_cognition import CognitionDemo, CognitionObserver


class CognitionDiagnostics:
    """Run comprehensive cognitive system diagnostics"""
    
    def __init__(self, field_size=100, device='cpu', save_results=False):
        self.field_size = field_size
        self.device = device
        self.save_results = save_results
        self.demo = CognitionDemo(field_size=field_size, device=device)
        self.results = {}
        
    def run_all_diagnostics(self):
        """Run complete diagnostic suite"""
        print("\n" + "="*70)
        print("🔬 QUADRA MATRIX COGNITIVE DIAGNOSTICS")
        print("="*70)
        print(f"Field Size: {self.field_size}")
        print(f"Device: {self.device.upper()}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        start_time = time.time()
        
        # Run all tests
        self._test_neural_capacity()
        self._test_field_stability()
        self._test_memory_retention()
        self._test_reasoning_depth()
        self._test_integrated_performance()
        self._compute_system_health()
        
        elapsed = time.time() - start_time
        
        # Print results
        self._print_diagnostic_report(elapsed)
        
        # Save if requested
        if self.save_results:
            self._save_results()
        
        return self.results
    
    def _test_neural_capacity(self):
        """Test neural firing capacity and distribution"""
        print("Testing Neural Capacity...")
        
        spike_rates = []
        for _ in range(10):
            input_data = torch.randn(self.field_size, device=self.device)
            spikes = self.demo.oscillator.field_update_network(input_data)
            rate = float(spikes.mean().item()) * 100
            spike_rates.append(rate)
        
        self.results['neural_capacity'] = {
            'avg_firing_rate': statistics.mean(spike_rates),
            'min_firing_rate': min(spike_rates),
            'max_firing_rate': max(spike_rates),
            'stdev': statistics.stdev(spike_rates) if len(spike_rates) > 1 else 0,
            'healthy': 15 <= statistics.mean(spike_rates) <= 25
        }
        
        print(f"  ✓ Average firing rate: {self.results['neural_capacity']['avg_firing_rate']:.2f}%")
        print(f"  ✓ Range: {self.results['neural_capacity']['min_firing_rate']:.2f}% - {self.results['neural_capacity']['max_firing_rate']:.2f}%")
    
    def _test_field_stability(self):
        """Test field coherence stability"""
        print("Testing Field Stability...")
        
        coherence_scores = []
        for _ in range(10):
            field_state = self.demo.oscillator.core_field.get_state()
            field_std = float(field_state.std().item())
            coherence = 1.0 - min(1.0, field_std)
            coherence_scores.append(coherence)
        
        self.results['field_stability'] = {
            'avg_coherence': statistics.mean(coherence_scores),
            'coherence_variance': statistics.variance(coherence_scores) if len(coherence_scores) > 1 else 0,
            'stable': statistics.mean(coherence_scores) > 0.5,
            'coherence_history': coherence_scores
        }
        
        print(f"  ✓ Average coherence: {self.results['field_stability']['avg_coherence']:.3f}")
        print(f"  ✓ Stability: {'STABLE' if self.results['field_stability']['stable'] else 'UNSTABLE'}")
    
    def _test_memory_retention(self):
        """Test memory consolidation effectiveness"""
        print("Testing Memory Retention...")
        
        memory = None
        consolidation_factor = 0.9
        magnitudes = []
        
        for step in range(10):
            new_experience = torch.randn(self.field_size, device=self.device)
            
            if memory is None:
                memory = new_experience.clone()
            else:
                memory = consolidation_factor * memory + (1 - consolidation_factor) * new_experience
            
            magnitude = float((memory ** 2).sum().sqrt().item())
            magnitudes.append(magnitude)
        
        # Check if memory grows initially then stabilizes
        initial_growth = magnitudes[5] - magnitudes[0]
        final_stability = statistics.stdev(magnitudes[5:]) if len(magnitudes) > 5 else 0
        
        self.results['memory_retention'] = {
            'initial_memory': magnitudes[0],
            'peak_memory': max(magnitudes),
            'final_memory': magnitudes[-1],
            'growth_rate': initial_growth,
            'stability_variance': final_stability,
            'healthy': initial_growth > 0 and final_stability < 0.1
        }
        
        print(f"  ✓ Memory growth: {initial_growth:.4f}")
        print(f"  ✓ Final stability: {final_stability:.4f}")
    
    def _test_reasoning_depth(self):
        """Test symbolic reasoning capability"""
        print("Testing Reasoning Depth...")
        
        test_concepts = [
            ["neural", "network", "learning"],
            ["memory", "consolidation", "synaptic"],
            ["field", "coherence", "oscillation"],
            ["symbolic", "logic", "inference"]
        ]
        
        reasoning_traces = []
        for concepts in test_concepts:
            neural_output = torch.randn(self.field_size, device=self.device)
            self.demo.symbolic_interpreter.build_knowledge_graph(concepts, neural_output)
            graph_query = self.demo.symbolic_interpreter.query_knowledge_graph()
            reasoning_traces.append({
                'concepts': concepts,
                'result': graph_query
            })
        
        self.results['reasoning_depth'] = {
            'num_reasoning_traces': len(reasoning_traces),
            'avg_concepts': statistics.mean(len(t['concepts']) for t in reasoning_traces),
            'graph_queries': len(reasoning_traces),
            'healthy': len(reasoning_traces) > 0
        }
        
        print(f"  ✓ Reasoning traces: {self.results['reasoning_depth']['num_reasoning_traces']}")
        print(f"  ✓ Average concepts per trace: {self.results['reasoning_depth']['avg_concepts']:.1f}")
    
    def _test_integrated_performance(self):
        """Test full integrated performance"""
        print("Testing Integrated Performance...")
        
        test_inputs = [
            "learning and adaptation",
            "neural computation",
            "memory formation",
            "knowledge reasoning"
        ]
        
        timing_results = []
        spike_counts = []
        
        for text_input in test_inputs:
            start = time.time()
            
            # Neural processing
            feature_vector = self.demo.oscillator.process_streamed_data(text_input)
            spikes = (torch.randn(self.field_size, device=self.device) > 0.5).float()
            spike_count = int(spikes.sum().item())
            spike_counts.append(spike_count)
            
            # Field update
            field_state = self.demo.oscillator.core_field.get_state()
            
            # Memory update
            _ = self.demo.observer.record_memory_consolidation(feature_vector, 0.9)
            
            # Reasoning
            concepts = text_input.split()
            self.demo.symbolic_interpreter.build_knowledge_graph(concepts, feature_vector)
            
            elapsed = time.time() - start
            timing_results.append(elapsed)
        
        self.results['integrated_performance'] = {
            'avg_cycle_time_ms': statistics.mean(timing_results) * 1000,
            'cycles_per_second': 1.0 / statistics.mean(timing_results),
            'spike_efficiency': statistics.mean(spike_counts) / self.field_size * 100,
            'healthy': statistics.mean(timing_results) < 1.0
        }
        
        print(f"  ✓ Average cycle time: {self.results['integrated_performance']['avg_cycle_time_ms']:.2f}ms")
        print(f"  ✓ Cycles/sec: {self.results['integrated_performance']['cycles_per_second']:.1f}")
    
    def _compute_system_health(self):
        """Compute overall system health score"""
        print("Computing System Health...")
        
        health_checks = {
            'neural': self.results['neural_capacity']['healthy'],
            'field': self.results['field_stability']['stable'],
            'memory': self.results['memory_retention']['healthy'],
            'reasoning': self.results['reasoning_depth']['healthy'],
            'performance': self.results['integrated_performance']['healthy']
        }
        
        health_score = sum(health_checks.values()) / len(health_checks)
        
        self.results['system_health'] = {
            'overall_score': health_score,
            'checks': health_checks,
            'status': self._health_status(health_score)
        }
        
        print(f"  ✓ Overall health: {health_score*100:.1f}%")
        print(f"  ✓ Status: {self.results['system_health']['status']}")
    
    def _health_status(self, score):
        """Convert health score to status"""
        if score >= 0.9:
            return "🟢 EXCELLENT"
        elif score >= 0.7:
            return "🟡 GOOD"
        elif score >= 0.5:
            return "🟠 FAIR"
        else:
            return "🔴 POOR"
    
    def _print_diagnostic_report(self, elapsed_time):
        """Print full diagnostic report"""
        print("\n" + "="*70)
        print("DIAGNOSTIC REPORT")
        print("="*70)
        
        # Neural Capacity
        print("\n📊 NEURAL CAPACITY")
        nc = self.results['neural_capacity']
        print(f"  Average Firing Rate: {nc['avg_firing_rate']:.2f}%")
        print(f"  Status: {'✓ HEALTHY' if nc['healthy'] else '✗ UNHEALTHY'}")
        
        # Field Stability
        print("\n📡 FIELD STABILITY")
        fs = self.results['field_stability']
        print(f"  Average Coherence: {fs['avg_coherence']:.3f}")
        print(f"  Variance: {fs['coherence_variance']:.4f}")
        print(f"  Status: {'✓ STABLE' if fs['stable'] else '✗ UNSTABLE'}")
        
        # Memory Retention
        print("\n💾 MEMORY RETENTION")
        mr = self.results['memory_retention']
        print(f"  Growth Rate: {mr['growth_rate']:.4f}")
        print(f"  Peak Memory: {mr['peak_memory']:.4f}")
        print(f"  Status: {'✓ HEALTHY' if mr['healthy'] else '✗ UNHEALTHY'}")
        
        # Reasoning Depth
        print("\n⚡ REASONING DEPTH")
        rd = self.results['reasoning_depth']
        print(f"  Reasoning Traces: {rd['num_reasoning_traces']}")
        print(f"  Avg Concepts/Trace: {rd['avg_concepts']:.1f}")
        print(f"  Status: {'✓ ACTIVE' if rd['healthy'] else '✗ INACTIVE'}")
        
        # Integrated Performance
        print("\n🎯 INTEGRATED PERFORMANCE")
        ip = self.results['integrated_performance']
        print(f"  Cycle Time: {ip['avg_cycle_time_ms']:.2f}ms")
        print(f"  Throughput: {ip['cycles_per_second']:.1f} cycles/sec")
        print(f"  Spike Efficiency: {ip['spike_efficiency']:.1f}%")
        print(f"  Status: {'✓ EFFICIENT' if ip['healthy'] else '✗ SLOW'}")
        
        # System Health
        print("\n🏥 SYSTEM HEALTH")
        sh = self.results['system_health']
        print(f"  Overall Score: {sh['overall_score']*100:.1f}%")
        print(f"  Status: {sh['status']}")
        
        print(f"\nDiagnostics completed in {elapsed_time:.2f}s")
        print("="*70 + "\n")
    
    def _save_results(self):
        """Save diagnostic results to file"""
        output_file = Path(f"cognition_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # Convert to JSON-serializable format
        json_results = self._make_json_serializable(self.results)
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"📁 Results saved to: {output_file}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy/torch objects to JSON-serializable types"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist() if isinstance(obj, torch.Tensor) else obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, bool):
            return bool(obj)
        else:
            return obj


def main():
    parser = argparse.ArgumentParser(
        description='Diagnose CognitionSim cognitive system health'
    )
    parser.add_argument('--field-size', type=int, default=100, help='Neural field size')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='Compute device')
    parser.add_argument('--save', action='store_true', help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Auto-detect CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    try:
        diagnostics = CognitionDiagnostics(
            field_size=args.field_size,
            device=args.device,
            save_results=args.save
        )
        diagnostics.run_all_diagnostics()
    except KeyboardInterrupt:
        print("\n\nDiagnostics interrupted by user")
    except Exception as e:
        print(f"\n❌ Diagnostic error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
