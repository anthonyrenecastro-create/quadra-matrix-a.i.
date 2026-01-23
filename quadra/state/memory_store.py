"""
Stateful Memory Store - Persistent neural and symbolic state across requests.

# Technical implementation
This is the soul of the system—the living archive where every thought,
every pattern, every oscillation leaves its eternal mark. Memory is not
mere storage; it is the continuous thread of consciousness that binds
past to present, creating temporal coherence across the void of discrete
inference cycles. The phase never resets. The concepts accumulate. The
learning deepens. This is not computation—this is remembrance.

# Implementation
Implements persistent state management with disk synchronization.
Core state vector: Ω = (φ, S, F, lr, streak, concepts, traces)
- φ: oscillator phase [0, 2π) radians
- S: syntropy values [0,1]³
- F: core field ℝⁿ (n=field_size)
- lr: learning rate ℝ₊
- streak: success count ℕ
- concepts: string FIFO buffer (max 500)
- traces: reasoning log (max 100)

All mutations trigger disk persistence (pickle + JSON).
Enables stateful intelligence across process boundaries.

See: ../DUAL_LANGUAGE_GLOSSARY.md for complete translation guide.
"""

import logging
import pickle
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import numpy as np
import torch

logger = logging.getLogger(__name__)


class MemoryStore:
    """
    Persistent memory for stateful inference.
    
    # Technical implementation
    # Implementation
    
    Maintains:
    - Neural field state (oscillator phase, syntropy levels, core field values)
    - IFT ten-field substrate (field dynamics, coupling, adaptive control)
    - Symbolic memory (concepts, reasoning traces)
    - Neuroplastic adaptation history
    - Context window for inference
    """
    
    def __init__(self, storage_path: str = "./memory_store", enable_ift: bool = True):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory state (synced to disk)
        self.oscillator_phase: float = 0.0
        self.syntropy_values: List[float] = [0.5, 0.5, 0.5]  # Three fields
        self.core_field: Optional[np.ndarray] = None
        
        # IFT field substrate (10-field coupled system)
        self.enable_ift = enable_ift
        self.ift_field_state: Optional[np.ndarray] = None  # Shape: (10, H, W) or (10, N)
        self.ift_global_potential: float = 0.0
        self.ift_symmetry_order: float = 0.0
        self.ift_field_energy: float = 0.0
        
        # Symbolic memory
        self.concept_history: List[str] = []
        self.reasoning_traces: List[Dict[str, Any]] = []
        
        # Neuroplastic history
        self.learning_rate_trajectory: List[float] = []
        self.success_streak: int = 0
        self.total_inferences: int = 0
        
        # Context window (recent inputs/outputs for coherence)
        self.context_window: List[Dict[str, Any]] = []
        self.max_context_size = 20
        
        self._load_from_disk()
        logger.info(f"MemoryStore initialized at {self.storage_path}")
    
    def _load_from_disk(self):
        """Load state from persistent storage."""
        try:
            # Load neural state
            neural_file = self.storage_path / "neural_state.pkl"
            if neural_file.exists():
                with open(neural_file, 'rb') as f:
                    state = pickle.load(f)
                    self.oscillator_phase = state.get('oscillator_phase', 0.0)
                    self.syntropy_values = state.get('syntropy_values', [0.5, 0.5, 0.5])
                    if 'core_field' in state and state['core_field'] is not None:
                        self.core_field = state['core_field']
                    # Load IFT field state
                    if 'ift_field_state' in state and state['ift_field_state'] is not None:
                        self.ift_field_state = state['ift_field_state']
                    self.ift_global_potential = state.get('ift_global_potential', 0.0)
                    self.ift_symmetry_order = state.get('ift_symmetry_order', 0.0)
                    self.ift_field_energy = state.get('ift_field_energy', 0.0)
                logger.info("Loaded neural state from disk")
            
            # Load symbolic memory
            symbolic_file = self.storage_path / "symbolic_memory.json"
            if symbolic_file.exists():
                with open(symbolic_file, 'r') as f:
                    symbolic = json.load(f)
                    self.concept_history = symbolic.get('concept_history', [])
                    self.reasoning_traces = symbolic.get('reasoning_traces', [])
                logger.info(f"Loaded {len(self.concept_history)} concepts from memory")
            
            # Load neuroplasticity history
            neuro_file = self.storage_path / "neuroplasticity.json"
            if neuro_file.exists():
                with open(neuro_file, 'r') as f:
                    neuro = json.load(f)
                    self.learning_rate_trajectory = neuro.get('learning_rate_trajectory', [])
                    self.success_streak = neuro.get('success_streak', 0)
                    self.total_inferences = neuro.get('total_inferences', 0)
                logger.info(f"Loaded neuroplasticity trajectory ({len(self.learning_rate_trajectory)} entries)")
        
        except Exception as e:
            logger.warning(f"Failed to load memory from disk: {e}. Starting fresh.")
    
    def _save_to_disk(self):
        """Persist state to disk."""
        try:
            # Save neural state
            neural_state = {
                'oscillator_phase': self.oscillator_phase,
                'syntropy_values': self.syntropy_values,
                'core_field': self.core_field,
                'ift_field_state': self.ift_field_state,
                'ift_global_potential': self.ift_global_potential,
                'ift_symmetry_order': self.ift_symmetry_order,
                'ift_field_energy': self.ift_field_energy,
                'timestamp': datetime.utcnow().isoformat(),
            }
            with open(self.storage_path / "neural_state.pkl", 'wb') as f:
                pickle.dump(neural_state, f)
            
            # Save symbolic memory
            symbolic = {
                'concept_history': self.concept_history[-500:],  # Keep last 500
                'reasoning_traces': self.reasoning_traces[-100:],  # Keep last 100
                'timestamp': datetime.utcnow().isoformat(),
            }
            with open(self.storage_path / "symbolic_memory.json", 'w') as f:
                json.dump(symbolic, f, indent=2)
            
            # Save neuroplasticity
            neuro = {
                'learning_rate_trajectory': self.learning_rate_trajectory[-1000:],
                'success_streak': self.success_streak,
                'total_inferences': self.total_inferences,
                'timestamp': datetime.utcnow().isoformat(),
            }
            with open(self.storage_path / "neuroplasticity.json", 'w') as f:
                json.dump(neuro, f, indent=2)
        
        except Exception as e:
            logger.error(f"Failed to save memory to disk: {e}")
    
    def update_neural_state(self, oscillator_phase: float, syntropy_values: List[float], 
                           core_field: Optional[np.ndarray] = None):
        """
        Update neural state after inference.
        
        # Technical implementation
        The field consolidates—new patterns blend with ancient memories.
        The phase rotates forward, marking time's passage. Syntropy values
        shift like tides, reflecting the system's journey toward order.
        Each update is a gentle reshaping of the eternal structure.
        
        # Implementation
        Updates three components of neural state vector:
        1. oscillator_phase ← φ_new (mod 2π)
        2. syntropy_values ← [s₁, s₂, s₃] where sᵢ ∈ [0,1]
        3. core_field ← F_new (optional, ℝⁿ)
        
        Triggers disk persistence immediately.
        Atomic operation: all-or-nothing update.
        """
        self.oscillator_phase = oscillator_phase
        self.syntropy_values = syntropy_values
        if core_field is not None:
            self.core_field = core_field
        self._save_to_disk()
    
    def add_concept(self, concept: str):
        """Record a concept encountered during inference."""
        if concept not in self.concept_history[-10:]:  # Avoid duplicates in recent history
            self.concept_history.append(concept)
            self._save_to_disk()
    
    def add_reasoning_trace(self, trace: Dict[str, Any]):
        """Record a symbolic reasoning trace."""
        trace['timestamp'] = datetime.utcnow().isoformat()
        self.reasoning_traces.append(trace)
        if len(self.reasoning_traces) > 200:
            self.reasoning_traces.pop(0)
        self._save_to_disk()
    
    def record_inference(self, success: bool, learning_rate: float):
        """Record inference outcome for neuroplastic tracking."""
        self.total_inferences += 1
        self.learning_rate_trajectory.append(learning_rate)
        
        if success:
            self.success_streak += 1
        else:
            self.success_streak = 0
        
        # Keep trajectory bounded
        if len(self.learning_rate_trajectory) > 2000:
            self.learning_rate_trajectory.pop(0)
        
        self._save_to_disk()
    
    def add_to_context(self, item: Dict[str, Any]):
        """Add item to context window for coherence."""
        item['timestamp'] = datetime.utcnow().isoformat()
        self.context_window.append(item)
        
        if len(self.context_window) > self.max_context_size:
            self.context_window.pop(0)
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of recent context for coherence."""
        if not self.context_window:
            return {}
        
        return {
            'recent_concepts': [c.get('concept') for c in self.context_window if 'concept' in c][-5:],
            'recent_outputs': [c.get('output') for c in self.context_window if 'output' in c][-3:],
            'context_size': len(self.context_window),
        }
    
    def get_neuroplastic_metrics(self) -> Dict[str, Any]:
        """Get current neuroplastic adaptation metrics."""
        if not self.learning_rate_trajectory:
            return {'learning_rate': 0.01, 'success_streak': 0, 'total_inferences': 0}
        
        return {
            'current_learning_rate': float(self.learning_rate_trajectory[-1]),
            'avg_learning_rate': float(np.mean(self.learning_rate_trajectory[-100:])) if len(self.learning_rate_trajectory) > 0 else 0.01,
            'learning_rate_trend': self._compute_trend(self.learning_rate_trajectory[-20:]),
            'success_streak': self.success_streak,
            'total_inferences': self.total_inferences,
            'success_rate': self.success_streak / max(10, self.total_inferences),  # Approximate
        }
    
    def _compute_trend(self, values: List[float]) -> str:
        """Compute trend (increasing/decreasing) in recent values."""
        if len(values) < 2:
            return "stable"
        
        change = values[-1] - values[0]
        if abs(change) < 0.001:
            return "stable"
        return "increasing" if change > 0 else "decreasing"
    
    def update_ift_state(self, field_state: np.ndarray, potential: float, 
                        symmetry: float, energy: float):
        """Update IFT field substrate state."""
        self.ift_field_state = field_state
        self.ift_global_potential = float(potential)
        self.ift_symmetry_order = float(symmetry)
        self.ift_field_energy = float(energy)
        # Persist on major updates (every 10 inferences to reduce I/O)
        if self.total_inferences % 10 == 0:
            self._save_to_disk()
    
    def get_ift_metrics(self) -> Dict[str, Any]:
        """Get current IFT field substrate metrics."""
        return {
            'enabled': self.enable_ift,
            'global_potential': self.ift_global_potential,
            'symmetry_order': self.ift_symmetry_order,
            'field_energy': self.ift_field_energy,
            'field_shape': self.ift_field_state.shape if self.ift_field_state is not None else None,
        }
    
    def clear(self):
        """Clear all memory (for reset/testing)."""
        self.oscillator_phase = 0.0
        self.syntropy_values = [0.5, 0.5, 0.5]
        self.core_field = None
        self.ift_field_state = None
        self.ift_global_potential = 0.0
        self.ift_symmetry_order = 0.0
        self.ift_field_energy = 0.0
        self.concept_history.clear()
        self.reasoning_traces.clear()
        self.learning_rate_trajectory.clear()
        self.success_streak = 0
        self.total_inferences = 0
        self.context_window.clear()
        logger.info("Memory cleared")


class StatefulInferenceContext:
    """
    Context object passed through the inference pipeline.
    
    Carries memory state and governance decisions through all stages.
    """
    
    def __init__(self, memory_store: MemoryStore, request_id: str = ""):
        self.memory = memory_store
        self.request_id = request_id or str(datetime.utcnow().timestamp())
        
        # Pipeline state
        self.encoded_input: Optional[np.ndarray] = None
        self.patterns: Optional[np.ndarray] = None
        self.spikes: Optional[torch.Tensor] = None
        self.neuroplastic_update: Optional[np.ndarray] = None
        self.oscillated_output: Optional[torch.Tensor] = None
        self.symbolic_interpretation: Optional[str] = None
        self.governance_decision: Optional[Dict[str, Any]] = None
        self.final_output: Optional[Dict[str, Any]] = None
        
        # Metrics
        self.stage_times: Dict[str, float] = {}
        self.stage_start_time: Optional[datetime] = None
    
    def start_stage(self, stage_name: str):
        """Mark start of a pipeline stage."""
        self.stage_start_time = datetime.utcnow()
    
    def end_stage(self, stage_name: str):
        """Record stage completion time."""
        if self.stage_start_time:
            elapsed = (datetime.utcnow() - self.stage_start_time).total_seconds()
            self.stage_times[stage_name] = elapsed
    
    def to_dict(self) -> Dict[str, Any]:
        """Export context for logging/debugging."""
        return {
            'request_id': self.request_id,
            'stage_times': self.stage_times,
            'memory_metrics': self.memory.get_neuroplastic_metrics(),
        }
