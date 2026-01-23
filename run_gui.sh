#!/bin/bash
# CognitionSim Cognition - GUI Launcher
# Run visual demonstration of cognitive processes

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     🧠 QUADRA MATRIX COGNITION VISUALIZER                      ║"
echo "║     Interactive GUI for observing cognitive processes          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if GUI is available
if ! python -c "import tkinter; import matplotlib" 2>/dev/null; then
    echo "⚠️  GUI dependencies not found. Installing..."
    pip install matplotlib -q
fi

echo "Launching Cognition Visualizer..."
echo ""
echo "Features:"
echo "  🧠 Real-time neural spiking visualization"
echo "  📡 Live field coherence tracking"
echo "  💾 Memory consolidation progress"
echo "  ⚡ Symbolic reasoning process tracking"
echo ""
echo "Controls:"
echo "  ▶ Run Full Demo      - Run complete cognitive demonstration"
echo "  🧠 Neural Spiking    - Individual neural firing demo"
echo "  📡 Field Coherence   - Field stability demo"
echo "  💾 Memory            - Memory consolidation demo"
echo "  ⚡ Reasoning         - Symbolic reasoning demo"
echo "  🔄 Reset             - Clear all data"
echo "  ⏹ Stop              - Stop current demo"
echo ""

python gui_cognition.py
