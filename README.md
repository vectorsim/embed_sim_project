# EmbedSim

**An open-source Python framework for simulation and C code generation targeting embedded control systems.**

EmbedSim is a free, accessible alternative to MATLAB/Simulink, designed for embedded control engineers who need rigorous signal-flow simulation without proprietary toolchain lock-in. Built on a block-diagram paradigm with a clean Python API, it bridges the gap between algorithm development and production C code for MCU targets.

---

## What It Does

EmbedSim lets you build, simulate, and generate embedded C code from the same model — using nothing but Python and open-source tools.

- **Block-diagram simulation** — wire up control blocks using the `>>` operator and simulate with RK4 or Euler integrators
- **Dual Python/C backend** — every block runs in pure Python for rapid prototyping, or against a compiled Cython/C backend for performance and fidelity
- **Automatic C code generation** — emit `embedsim_loop.c` / `embedsim_loop.h` directly from your simulation graph for deployment on embedded targets
- **FMU co-simulation** — integrate FMI 2.0 `.fmu` models as first-class blocks alongside native Python blocks
- **Algebraic loop handling** — DFS-based topological execution with `LoopBreaker` / `VectorDelayEnhanced` for feedback path management

---

## Target Platforms

- **Infineon AURIX TriCore** (TC3xx series) — primary MCU target
- **ARM Cortex-M4** — secondary target
- MISRA C:2012 and ASIL-D compatible code generation

---

## Example Use Cases

- PMSM Field-Oriented Control (FOC) with SMC / SMO / PWM blocks
- Clarke / Park coordinate transforms with Cython wrappers
- RLC circuit LQR state-space controllers
- Speed PI controllers with RK4-compatible integrators
- Digital twin co-simulation via FMU

---

## Architecture

```
VectorBlock          — base class for all simulation blocks
VectorSignal         — typed signal carrying float32 data between blocks
VectorSim            — simulation runner (topological sort + integrator)
PYXInspector         — introspects .pyx files to auto-populate block attributes
CodeGenerator        — walks the block graph and emits embedsim_loop.c/.h
FMUBlock             — wraps FMI 2.0 .fmu models as native blocks
ScriptBlock          — AST-based C code generation for inline logic
```

---

## Why EmbedSim?

| | MATLAB/Simulink | EmbedSim |
|---|---|---|
| Cost | Expensive licensing | Free / open-source |
| Language | Proprietary | Python + C |
| Code generation | Embedded Coder (paid) | Built-in |
| Extensibility | Limited | Full Python ecosystem |
| Version control | Difficult (.slx) | Plain text, git-friendly |

---

## Status

EmbedSim is under active development. Core simulation, Cython backend, and C code generation are functional. Contributions and feedback welcome.

---

*Built by engineers, for engineers. Democratising embedded control development.*
