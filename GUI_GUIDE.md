# Quadra Matrix Cognition Visualizer - GUI Guide

## Quick Start

```bash
# Option 1: Run the bash script (recommended)
bash run_gui.sh

# Option 2: Run directly with Python
python gui_cognition.py
```

## What You'll See

The GUI displays four synchronized real-time visualizations:

### 1. 🧠 Neural Firing Rate
- **Left plot, top**: Shows neurons firing over time
- **Healthy range**: 15-20% (sparse, efficient)
- **What to watch**: Regular oscillations indicate healthy spiking patterns
- **Updates**: Every neural processing step

### 2. 📡 Field Coherence
- **Right plot, top**: Shows global field organization
- **Range**: 0 (chaotic) to 1 (perfectly synchronized)
- **Healthy range**: 0.4-0.8 (organized)
- **What to watch**: High coherence shows system is focused/solving
- **Updates**: Every field evolution step

### 3. 💾 Memory Consolidation
- **Left plot, bottom**: Shows memory magnitude over time
- **Pattern**: Rapid initial growth, then gradual consolidation
- **What to watch**: Exponential consolidation pattern (EMA)
- **Meaning**: System is learning and integrating experiences
- **Updates**: Every memory update

### 4. ⚡ Symbolic Reasoning
- **Right plot, bottom**: Shows reasoning process count
- **What to watch**: Increasing count shows deeper reasoning
- **Meaning**: Knowledge graphs expanding with new concepts
- **Updates**: Every reasoning step

## Control Panel

### Buttons

| Button | Purpose | What It Does |
|--------|---------|-------------|
| **▶ Run Full Demo** | Complete demonstration | Runs all 4 cognitive processes in sequence (neural → coherence → memory → reasoning) |
| **🧠 Neural Spiking** | Individual demo | Show only neural firing dynamics |
| **📡 Field Coherence** | Individual demo | Show only field organization |
| **💾 Memory** | Individual demo | Show only memory consolidation |
| **⚡ Reasoning** | Individual demo | Show only symbolic reasoning |
| **🔄 Reset** | Clear data | Clear all buffers and plots |
| **⏹ Stop** | Stop demo | Halt current demonstration |

### Live Metrics Display

At the top, four metrics update in real-time:

- **Neural Firing Rate**: Current percentage of neurons firing (%)
- **Field Coherence**: Current field organization (0-1)
- **Memory Magnitude**: Current memory size
- **Reasoning Count**: Number of reasoning processes completed

## Understanding the Visualizations

### Neural Firing Pattern
```
Expected: 
  ▄▅▆█▆▅▄  ← Oscillating pattern, 15-20% average
  
Problems:
  ▁▁▁▁▁▁▁  ← Too sparse (< 5%)
  █████████ ← Too dense (> 50%)
```

### Field Coherence Pattern
```
Expected during solving:
  ▁▂▃▄▅▆ ← Rises as system solves
  
Expected during processing:
  ▄▃▂▁▂▃ ← Dips while exploring
```

### Memory Consolidation Pattern
```
Expected:
  ▁▂▄▆█▇▇▇▇ ← Rapid rise, then plateau (exponential)
  
This shows: New experiences integrate with old memories
```

### Reasoning Pattern
```
Expected:
  ▁▁▂▂▃▃▄▄ ← Steady increase as more concepts added
  
This shows: Knowledge graph expanding
```

## Demo Modes

### Full Demo (▶ Run Full Demo)
Runs all four cognitive processes in sequence:

1. **Neural Spiking** (3 seconds)
   - Watch neurons fire in the first plot
   - Observe firing rate metric

2. **Field Coherence** (3 seconds)
   - Watch field organize in second plot
   - Observe coherence metric

3. **Memory Consolidation** (3 seconds)
   - Watch memory grow and stabilize in third plot
   - Observe consolidation pattern

4. **Symbolic Reasoning** (3 seconds)
   - Watch reasoning count increase in fourth plot
   - Observe knowledge graph expansion

**Total time**: ~15 seconds

### Individual Demos
Click any of the individual buttons to see just that process:
- **Neural Spiking**: 2 seconds
- **Field Coherence**: 2 seconds
- **Memory**: 2 seconds
- **Reasoning**: 2 seconds

## Interpretation Guide

### What Good Metrics Look Like

| Metric | Good Value | Indicator |
|--------|-----------|-----------|
| **Firing Rate** | 15-20% | System is efficient |
| **Coherence** | 0.4-0.8 | System is organized |
| **Memory** | Growing then stable | Learning happening |
| **Reasoning** | Increasing | Deeper understanding |

### What Problem Metrics Look Like

| Metric | Bad Value | Problem |
|--------|-----------|---------|
| **Firing Rate** | < 5% | Dead neurons |
| **Firing Rate** | > 40% | Over-stimulation |
| **Coherence** | < 0.3 | System confused |
| **Coherence** | > 0.95 | System stuck |
| **Memory** | Flat | Not learning |
| **Memory** | Crashing | Instability |
| **Reasoning** | Flat | No learning |

## Tips for Best Visualization

1. **Start with Full Demo**: Get a sense of all processes together
2. **Then run Individual Demos**: Study each process in detail
3. **Watch the metrics**: They tell the story of cognition
4. **Reset between runs**: Click 🔄 to clear old data
5. **Look for patterns**: Each process has a characteristic pattern

## Common Questions

### Q: Why are firing rates so low?
**A:** That's correct! Spiking neural networks are sparse by design. 15-20% firing is efficient and biologically plausible.

### Q: The coherence keeps dropping - is something wrong?
**A:** No! During processing, coherence naturally drops as the system "thinks." It rises when it converges on a solution.

### Q: Memory keeps going down after rising - is that OK?
**A:** Yes! That's exponential moving average (EMA) at work. Memory magnitude stabilizes as experiences consolidate.

### Q: Why does reasoning count jump?
**A:** Each new concept added to the knowledge graph increases the count. This shows the system is building understanding.

### Q: Can I make the plots bigger?
**A:** Yes! Resize the window. The plots will scale automatically.

### Q: How long does each demo run?
**A:** Full demo: ~15 seconds. Individual demos: ~2 seconds each. Click ⏹ Stop to interrupt.

## Advanced Usage

### Modify Demo Parameters
Edit `gui_cognition.py` to change:
- `field_size`: Larger = more neurons, slower
- `num_iterations`: More iterations = longer demo
- `sleep_time`: Demo speed

### Add Custom Metrics
Edit the metrics section to track additional metrics like:
- Loss values
- Q-learning rewards
- Field resonance
- Gradient norms

### Save Plot Data
The plots update in real-time. You can:
1. Take screenshots during demos
2. Export data programmatically
3. Create custom analysis tools

## Performance Notes

- **CPU**: Smooth performance on all modern CPUs
- **GPU**: Automatic detection and use if available
- **Memory**: ~500MB typical usage
- **Latency**: < 100ms between updates

## Troubleshooting

### Window doesn't appear
```bash
# Check display settings
python gui_cognition.py
# If no window: May be display issue in remote/headless environment
```

### Plots don't update
- Click "Run Full Demo" to start
- Check that demo is actually running (status bar shows progress)
- Try "Reset" then run again

### Metrics show "--"
- Metrics only populate during active demonstration
- Run any demo to see values
- Individual demos show metrics specific to that process

### Slow performance
- Close other applications
- Use CPU-only mode (default)
- Reduce window size

## Next Steps

1. **See it work**: Run Full Demo to observe all processes
2. **Understand it**: Read [COGNITION_OBSERVATION_GUIDE.md](./COGNITION_OBSERVATION_GUIDE.md)
3. **Use it**: Integrate into your own applications
4. **Extend it**: Modify GUI to track custom metrics

## Related Resources

- [QUICK_COGNITION_START.md](./QUICK_COGNITION_START.md) - 5-minute intro
- [COGNITION_OBSERVATION_GUIDE.md](./COGNITION_OBSERVATION_GUIDE.md) - Detailed guide
- [demo_cognition.py](./demo_cognition.py) - Core demonstration code
- [COGNITIVE_MODEL.md](./COGNITIVE_MODEL.md) - Mathematical foundations

---

Enjoy exploring Quadra Matrix cognition visually! 🧠✨
