#!/usr/bin/env python3
"""
Quadra Matrix Cognition - Visual GUI Demonstration
Interactive graphical interface to observe cognitive processes in real-time.

Features:
- Real-time visualization of neural spiking
- Live field coherence tracking
- Memory consolidation progress
- Symbolic reasoning knowledge graph
- Multiple display modes
"""

import tkinter as tk
from tkinter import ttk, messagebox
import torch
import numpy as np
import threading
import time
from collections import deque
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import matplotlib.animation as animation
except ImportError:
    logger.error("matplotlib not found. Install with: pip install matplotlib")
    exit(1)

from demo_cognition import CognitionDemo


class CognitionGUI:
    """Interactive GUI for observing cognitive processes"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 Quadra Matrix Cognition Visualizer")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize demo
        self.demo = CognitionDemo(field_size=100, device='cpu')
        self.running = False
        
        # Data buffers
        self.spike_buffer = deque(maxlen=50)
        self.coherence_buffer = deque(maxlen=50)
        self.memory_buffer = deque(maxlen=50)
        self.time_buffer = deque(maxlen=50)
        self.current_time = 0
        
        # Current state
        self.neural_firing_rate = 0
        self.field_coherence = 0
        self.memory_magnitude = 0
        self.reasoning_count = 0
        
        self._create_ui()
        
    def _create_ui(self):
        """Create the user interface"""
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50')
        title_frame.pack(fill=tk.X, padx=0, pady=0)
        
        title_label = tk.Label(
            title_frame, 
            text="🧠 QUADRA MATRIX COGNITION VISUALIZER",
            font=('Arial', 18, 'bold'),
            fg='white',
            bg='#2c3e50',
            pady=10
        )
        title_label.pack()
        
        # Control Panel
        control_frame = tk.Frame(self.root, bg='#ecf0f1')
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.run_button = tk.Button(
            control_frame,
            text="▶ Run Full Demo",
            command=self.start_demo,
            font=('Arial', 12, 'bold'),
            bg='#27ae60',
            fg='white',
            padx=15,
            pady=8
        )
        self.run_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = tk.Button(
            control_frame,
            text="⏹ Stop",
            command=self.stop_demo,
            font=('Arial', 12, 'bold'),
            bg='#e74c3c',
            fg='white',
            padx=15,
            pady=8,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.neural_button = tk.Button(
            control_frame,
            text="🧠 Neural Spiking",
            command=lambda: self.run_single_demo('neural'),
            font=('Arial', 10),
            bg='#3498db',
            fg='white',
            padx=10,
            pady=6
        )
        self.neural_button.pack(side=tk.LEFT, padx=5)
        
        self.field_button = tk.Button(
            control_frame,
            text="📡 Field Coherence",
            command=lambda: self.run_single_demo('field'),
            font=('Arial', 10),
            bg='#9b59b6',
            fg='white',
            padx=10,
            pady=6
        )
        self.field_button.pack(side=tk.LEFT, padx=5)
        
        self.memory_button = tk.Button(
            control_frame,
            text="💾 Memory",
            command=lambda: self.run_single_demo('memory'),
            font=('Arial', 10),
            bg='#f39c12',
            fg='white',
            padx=10,
            pady=6
        )
        self.memory_button.pack(side=tk.LEFT, padx=5)
        
        self.reasoning_button = tk.Button(
            control_frame,
            text="⚡ Reasoning",
            command=lambda: self.run_single_demo('reasoning'),
            font=('Arial', 10),
            bg='#e67e22',
            fg='white',
            padx=10,
            pady=6
        )
        self.reasoning_button.pack(side=tk.LEFT, padx=5)
        
        self.reset_button = tk.Button(
            control_frame,
            text="🔄 Reset",
            command=self.reset_data,
            font=('Arial', 10),
            bg='#95a5a6',
            fg='white',
            padx=10,
            pady=6
        )
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
        # Metrics Panel
        metrics_frame = tk.LabelFrame(self.root, text="📊 Live Metrics", font=('Arial', 11, 'bold'), bg='#ecf0f1')
        metrics_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create metric displays
        self.metrics_labels = {}
        metrics = [
            ('Neural Firing Rate', 'firing_rate'),
            ('Field Coherence', 'coherence'),
            ('Memory Magnitude', 'memory'),
            ('Reasoning Count', 'reasoning')
        ]
        
        for i, (label, key) in enumerate(metrics):
            col = i % 2
            row = i // 2
            
            frame = tk.Frame(metrics_frame, bg='#ecf0f1')
            frame.grid(row=row, column=col, padx=15, pady=10, sticky='ew')
            
            tk.Label(frame, text=label + ":", font=('Arial', 10, 'bold'), bg='#ecf0f1').pack(side=tk.LEFT)
            label_widget = tk.Label(frame, text="--", font=('Arial', 12, 'bold'), fg='#2c3e50', bg='#ecf0f1')
            label_widget.pack(side=tk.LEFT, padx=10)
            
            self.metrics_labels[key] = label_widget
        
        # Visualization area
        plot_frame = tk.LabelFrame(self.root, text="🎨 Visualizations", font=('Arial', 11, 'bold'), bg='#ecf0f1')
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create figure with 4 subplots
        self.fig = Figure(figsize=(14, 6), dpi=100)
        self.fig.patch.set_facecolor('#f0f0f0')
        
        # Neural firing
        self.ax1 = self.fig.add_subplot(2, 2, 1)
        self.ax1.set_title('Neural Firing Rate', fontweight='bold')
        self.ax1.set_ylabel('Firing Rate (%)')
        self.ax1.set_ylim(0, 100)
        self.line1, = self.ax1.plot([], [], 'b-', linewidth=2, label='Firing Rate')
        self.ax1.legend(loc='upper left')
        self.ax1.grid(True, alpha=0.3)
        
        # Field coherence
        self.ax2 = self.fig.add_subplot(2, 2, 2)
        self.ax2.set_title('Field Coherence', fontweight='bold')
        self.ax2.set_ylabel('Coherence Score')
        self.ax2.set_ylim(0, 1)
        self.line2, = self.ax2.plot([], [], 'purple', linewidth=2, label='Coherence')
        self.ax2.legend(loc='upper left')
        self.ax2.grid(True, alpha=0.3)
        
        # Memory consolidation
        self.ax3 = self.fig.add_subplot(2, 2, 3)
        self.ax3.set_title('Memory Consolidation', fontweight='bold')
        self.ax3.set_ylabel('Memory Magnitude')
        self.line3, = self.ax3.plot([], [], 'orange', linewidth=2, label='Memory')
        self.ax3.legend(loc='upper left')
        self.ax3.grid(True, alpha=0.3)
        
        # Reasoning progress
        self.ax4 = self.fig.add_subplot(2, 2, 4)
        self.ax4.set_title('Reasoning Processes', fontweight='bold')
        self.ax4.set_ylabel('Count')
        self.line4, = self.ax4.plot([], [], 'red', linewidth=2, label='Reasoning')
        self.ax4.legend(loc='upper left')
        self.ax4.grid(True, alpha=0.3)
        
        # Embed matplotlib in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready | Click 'Run Full Demo' to start")
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=('Arial', 9),
            bg='#34495e',
            fg='white',
            pady=5
        )
        status_bar.pack(fill=tk.X)
    
    def update_metrics(self, firing_rate, coherence, memory, reasoning):
        """Update metric displays"""
        self.neural_firing_rate = firing_rate
        self.field_coherence = coherence
        self.memory_magnitude = memory
        self.reasoning_count = reasoning
        
        self.metrics_labels['firing_rate'].config(text=f"{firing_rate:.1f}%")
        self.metrics_labels['coherence'].config(text=f"{coherence:.3f}")
        self.metrics_labels['memory'].config(text=f"{memory:.4f}")
        self.metrics_labels['reasoning'].config(text=f"{reasoning}")
        
        # Add to buffers
        self.spike_buffer.append(firing_rate)
        self.coherence_buffer.append(coherence)
        self.memory_buffer.append(memory)
        self.time_buffer.append(self.current_time)
        self.current_time += 1
    
    def update_plots(self):
        """Update visualization plots"""
        # Neural firing
        if len(self.time_buffer) > 0:
            times = list(range(len(self.spike_buffer)))
            self.line1.set_data(times, list(self.spike_buffer))
            self.ax1.set_xlim(max(0, len(times) - 50), len(times))
            
            # Field coherence
            self.line2.set_data(times, list(self.coherence_buffer))
            self.ax2.set_xlim(max(0, len(times) - 50), len(times))
            
            # Memory
            self.line3.set_data(times, list(self.memory_buffer))
            self.ax3.set_xlim(max(0, len(times) - 50), len(times))
            
            # Reasoning
            reasoning_list = [self.reasoning_count] * len(self.time_buffer)
            self.line4.set_data(times, reasoning_list)
            self.ax4.set_xlim(max(0, len(times) - 50), len(times))
            
            self.fig.tight_layout()
            self.canvas.draw()
    
    def run_neural_demo(self):
        """Run neural spiking demonstration"""
        self.status_var.set("Running neural spiking demo...")
        self.root.update()
        
        for i in range(10):
            input_data = torch.randn(100, device='cpu')
            spikes1 = self.demo.oscillator.nn1(input_data)
            spikes2 = self.demo.oscillator.nn2(input_data)
            spikes3 = self.demo.oscillator.nn3(input_data)
            
            firing_rate = float(((spikes1.mean() + spikes2.mean() + spikes3.mean()) / 3).item()) * 100
            coherence = 1.0 - min(1.0, self.demo.oscillator.field.std().item())
            
            self.update_metrics(firing_rate, coherence, firing_rate / 10, i)
            self.update_plots()
            
            time.sleep(0.2)
            if not self.running:
                break
            self.root.update()
    
    def run_coherence_demo(self):
        """Run field coherence demonstration"""
        self.status_var.set("Running field coherence demo...")
        self.root.update()
        
        field_state = self.demo.oscillator.field.clone()
        
        for i in range(10):
            input_pert = torch.randn_like(field_state) * 0.1
            field_state = field_state + input_pert
            field_state = torch.clamp(field_state, -1, 1)
            
            coherence = 1.0 - min(1.0, field_state.std().item())
            firing_rate = float(torch.randn(1).item()) * 50 + 25
            
            self.update_metrics(firing_rate, coherence, coherence, i)
            self.update_plots()
            
            time.sleep(0.2)
            if not self.running:
                break
            self.root.update()
    
    def run_memory_demo(self):
        """Run memory consolidation demonstration"""
        self.status_var.set("Running memory consolidation demo...")
        self.root.update()
        
        memory = None
        
        for i in range(10):
            new_exp = torch.randn(100, device='cpu')
            
            if memory is None:
                memory = new_exp.clone()
            else:
                memory = 0.9 * memory + 0.1 * new_exp
            
            magnitude = float((memory ** 2).sum().sqrt().item())
            coherence = float((torch.randn(1).item()) * 0.3 + 0.7)
            
            self.update_metrics(magnitude / 10, coherence, magnitude / 10, i)
            self.update_plots()
            
            time.sleep(0.2)
            if not self.running:
                break
            self.root.update()
    
    def run_reasoning_demo(self):
        """Run symbolic reasoning demonstration"""
        self.status_var.set("Running symbolic reasoning demo...")
        self.root.update()
        
        for i in range(10):
            concepts = ["learning", "reasoning", "memory", "adaptation", "intelligence"]
            self.demo.symbolic_interpreter.build_knowledge_graph(
                concepts[:i % 5 + 1],
                torch.randn(100, device='cpu')
            )
            
            firing_rate = float(torch.randn(1).item()) * 30 + 20
            coherence = float(torch.randn(1).item()) * 0.2 + 0.8
            
            self.update_metrics(firing_rate, coherence, i / 10, i + 1)
            self.update_plots()
            
            time.sleep(0.3)
            if not self.running:
                break
            self.root.update()
    
    def run_full_demo(self):
        """Run complete demonstration"""
        self.status_var.set("Running full demonstration...")
        self.root.update()
        
        # Run each demo in sequence
        for demo_func, name in [
            (self.run_neural_demo, "Neural Spiking"),
            (self.run_coherence_demo, "Field Coherence"),
            (self.run_memory_demo, "Memory Consolidation"),
            (self.run_reasoning_demo, "Symbolic Reasoning")
        ]:
            if not self.running:
                break
            self.status_var.set(f"Running: {name}...")
            self.root.update()
            demo_func()
        
        if self.running:
            self.status_var.set("✅ Demonstration complete!")
    
    def start_demo(self):
        """Start the demonstration"""
        if self.running:
            return
        
        self.running = True
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.neural_button.config(state=tk.DISABLED)
        self.field_button.config(state=tk.DISABLED)
        self.memory_button.config(state=tk.DISABLED)
        self.reasoning_button.config(state=tk.DISABLED)
        
        thread = threading.Thread(target=self.run_full_demo, daemon=True)
        thread.start()
    
    def run_single_demo(self, demo_type):
        """Run a single demonstration type"""
        if self.running:
            return
        
        self.running = True
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.reset_data()
        
        demos = {
            'neural': self.run_neural_demo,
            'field': self.run_coherence_demo,
            'memory': self.run_memory_demo,
            'reasoning': self.run_reasoning_demo
        }
        
        thread = threading.Thread(target=demos[demo_type], daemon=True)
        thread.start()
    
    def stop_demo(self):
        """Stop the demonstration"""
        self.running = False
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.neural_button.config(state=tk.NORMAL)
        self.field_button.config(state=tk.NORMAL)
        self.memory_button.config(state=tk.NORMAL)
        self.reasoning_button.config(state=tk.NORMAL)
        self.status_var.set("Stopped | Ready for new demo")
    
    def reset_data(self):
        """Reset all data buffers"""
        self.spike_buffer.clear()
        self.coherence_buffer.clear()
        self.memory_buffer.clear()
        self.time_buffer.clear()
        self.current_time = 0
        
        self.line1.set_data([], [])
        self.line2.set_data([], [])
        self.line3.set_data([], [])
        self.line4.set_data([], [])
        
        self.canvas.draw()
        
        for label in self.metrics_labels.values():
            label.config(text="--")


def main():
    """Main entry point"""
    root = tk.Tk()
    gui = CognitionGUI(root)
    
    # Center window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'+{x}+{y}')
    
    root.mainloop()


if __name__ == "__main__":
    main()
