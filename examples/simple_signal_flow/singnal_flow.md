# User Note: Running the ControlForge/EmbedSim Examples

## Overview
This folder contains example scripts demonstrating the ControlForge/EmbedSim framework for block-diagram simulation targeting embedded systems.

## Prerequisites
- Python 3.7+ installed
- Required packages: `numpy`, `matplotlib`
- The `embedsim` package must be in your Python path (the examples handle this automatically)

## Available Examples

### 1. **example_two_sines_gain.py**
**Two sinusoidal sources + gain**
- Demonstrates basic signal generation and processing
- Two sine waves (5A@15Hz + 2A@15Hz, phase-shifted) → Sum → Gain (0.5) → Output
- Run: `python example_two_sines_gain.py`

### 2. **simple_signal_addition.py**
**Sine + Cosine + DC → Sum → Gain → Integrator**
- Shows more complex signal flow with dynamic blocks
- Demonstrates integration (stateful block)
- Run: `python simple_signal_addition.py`

### 3. **three_phase_source.py**
**Basic three-phase signal generation**
- Simplest example - just a source and sink
- Generates balanced 3-phase sine waves (120° separation)
- Run: `python three_phase_source.py`

### 4. **simple_signal-flow.cmd** (Windows only)
Batch script to run examples from a menu
- Double-click or run in Command Prompt

## Key Concepts Demonstrated

**Block Types:**
- **Sources**: `SinusoidalGenerator`, `ThreePhaseGenerator`, `VectorConstant`
- **Processing**: `VectorSum`, `VectorGain`
- **Dynamic**: `VectorIntegrator` (has state)
- **Sink**: `VectorEnd` (records data)

**Connections:** Use `>>` operator: `source >> gain >> sink`

**Simulation:** `EmbedSim` class with RK4 solver

**Visualization:** Each example plots results using matplotlib

## Troubleshooting

**"Module not found" errors:**
The examples automatically add the parent directory to sys.path. If you still get import errors:
- Ensure `embedsim` folder is in the same parent directory as the examples
- Or manually set `PYTHONPATH`

**No plot appears:**
- Check that matplotlib is installed: `pip install matplotlib`
- Some systems may need `plt.show()` configuration

**Simulation runs but no output:**
- Verify all blocks are properly connected with `>>`
- Check that sinks are included in `EmbedSim(sinks=[...])`
- Ensure scope.add() includes the signals you want to see