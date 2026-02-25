"""
RLC Digital Twin - vector_sim Framework Example
------------------------------------------------
Demonstrates a **digital twin of a series RLC circuit** driven by a 50 Hz
sine reference and regulated by a Python-side PI controller.  The plant is
provided as a Co-Simulation FMU (RLC_Sine_DigitalTwin_OM.fmu) compiled from
an OpenModelica model.

Circuit topology
----------------
    VS(t) ──► L ──► R ──► C ──► GND
                               │
                         VoltageSensor → Vout

Control loop (ASCII)
--------------------
    Vref ──►[+]──► [PI] ──► [Sat] ──► [FMU] ──► Vout
             ▲                                      │
             └──────────── [Delay] ─────────────────┘

    • The PI controller drives Vcontrol_python, which overrides the FMU's
      internal PI when usePythonControl = 1.
    • A one-step VectorDelay on Vout breaks the algebraic loop that would
      otherwise arise from the direct Vout → error feedback path.

Usage
-----
    python rlc_fmu.py
    (FMU must be present at  <script_dir>/modelica/RLC_Sine_DigitalTwin_OM.fmu)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# ---------------------------------------------------------------------------
# Path setup – works whether launched as a script (__file__ defined) or
# interactively in a notebook / REPL (__file__ not defined).
# ---------------------------------------------------------------------------
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    base_dir = Path(__file__).parent
except NameError:
    # Notebook / interactive session: fall back to current working directory
    _HERE = os.getcwd()
    base_dir = Path.cwd()

# Add the project root (two levels up) to sys.path so sim_core is importable
# without installing the package.
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..')))

from embedsim import (
    VectorSum,
    VectorGain,
    VectorSaturation,
    VectorDelay,
    VectorIntegrator,
    VectorEnd,
    EmbedSim,
    ODESolver,
    FMUBlock,
    DEFAULT_DTYPE,
    VectorBlock,
    VectorSignal
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# FMU location – OpenModelica-generated Co-Simulation FMU
FMU_PATH = base_dir / "modelica" / "RLC_Sine_DigitalTwin_OM.fmu"
if not FMU_PATH.exists():
    raise FileNotFoundError(
        f"FMU not found at: {FMU_PATH}\n"
        "Compile RLC_Sine_DigitalTwin.mo with OpenModelica and place the "
        "resulting .fmu in the 'modelica/' sub-folder."
    )

# --- Circuit parameters (must match the Modelica model defaults) ---
R = 10.0        # Series resistance  [Ω]
L = 10e-3       # Series inductance  [H]
C = 100e-6      # Series capacitance [F]

# Natural frequency  ω₀ = 1/√(LC) ≈ 1000 rad/s → f₀ ≈ 159 Hz
# Damping ratio      ζ  = R/(2) * √(C/L)        ≈ 1.58  (over-damped)

# --- Reference signal ---
VREF_AMPL = 10.0    # Sine amplitude [V]
FREQ      = 50.0    # Sine frequency [Hz]

# --- PI controller tuning ---
# Kp chosen empirically; Ti sets the integrator time constant.
# Ki = Kp / Ti ≈ 6250  → aggressive integral action; tightens steady-state
# error quickly at the cost of a slightly more oscillatory transient.
Kp = 20.0
Ti = 0.0032     # Integral time constant [s]

# --- Simulation parameters ---
T  = 0.1        # Total simulation time [s]  (5 full cycles at 50 Hz)
dt = 1e-5       # Time step [s] — Euler is stable for RLC with this step:
                # smallest time constant τ_min = 2L/R ≈ 2 ms  >> dt ✓

# ---------------------------------------------------------------------------
# Reference source block
# ---------------------------------------------------------------------------

class SineSource(VectorBlock):
    """
    Stateless signal source: outputs A·sin(2π·f·t) as a 1-element VectorSignal.

    Parameters
    ----------
    name      : str   – block label used in topology printouts and scope
    amplitude : float – peak amplitude [V]
    frequency : float – frequency [Hz]
    """

    def __init__(self, name: str, amplitude: float, frequency: float, use_c_backend: bool = False, dtype=None) -> None:
        _dtype = dtype if dtype is not None else DEFAULT_DTYPE
        super().__init__(name, use_c_backend=use_c_backend, dtype=_dtype)
        self.amplitude = amplitude
        self.frequency = frequency

    def compute_py(self, t: float, dt: float, input_values=None) -> VectorSignal:
        # No inputs needed – this is a pure source block
        value = self.amplitude * np.sin(2.0 * np.pi * self.frequency * t)
        self.output = VectorSignal([value], self.name)
        return self.output


# ---------------------------------------------------------------------------
# Block instantiation
# ---------------------------------------------------------------------------

# 1. Reference sine wave  →  Vref[0] = 10·sin(2π·50·t)
vref = SineSource("Vref", amplitude=VREF_AMPL, frequency=FREQ, use_c_backend=False)

# 2. One-step delay on Vout – breaks the algebraic loop.
#    initial=[0.0] means Vout(t=0⁻) = 0 V, i.e. capacitor starts uncharged.
#    The FMU output at step k feeds this block; the block feeds the error
#    summer at step k+1 (one dt later).
fb_delay = VectorDelay("Vout_delay", initial=[0.0])

# 3. Error summer: e = Vref − Vout_delayed
#    signs=[+1, -1] → first input is added, second is subtracted.
#    Wiring order MUST match signs order:
#      first  >> error  →  Vref      (sign +1)
#      second >> error  →  Vout_delay (sign -1)
error = VectorSum("error", signs=[1.0, -1.0], use_c_backend=False)
vref     >> error   # positive input  (+Vref)
fb_delay >> error   # negative input  (−Vout_delayed)

# 4. Proportional branch: P = Kp · e
prop = VectorGain("P_gain", gain=Kp, use_c_backend=False)
error >> prop

# 5. Integral branch: I = (Kp/Ti) · ∫e dt
#    The integrator state is updated by VectorSim at each step.
int_gain   = VectorGain("I_gain", gain=Kp / Ti)
integrator = VectorIntegrator("integrator", initial_state=[0.0], dim=1)
error >> int_gain >> integrator

# 6. PI output summer: u_PI = P + I
pi_out = VectorSum("PI_out", signs=[1.0, 1.0])
prop       >> pi_out
integrator >> pi_out

# 7. Anti-windup saturation: clamp control voltage to ±50 V
#    Prevents integrator windup when the FMU input rail is hit.
sat = VectorSaturation("Vctrl_sat", lower=-50.0, upper=50.0)
pi_out >> sat

# 8. RLC plant FMU (Co-Simulation)
#    Input  : Vcontrol_python – control voltage computed above
#    Output : Vout            – capacitor voltage (= plant output)
#    Setting usePythonControl=1.0 disables the FMU's built-in PI and hands
#    control authority entirely to the Python-side controller.
rlc_fmu = FMUBlock(
    name="RLC_plant",
    fmu_path=FMU_PATH,
    input_names=["Vcontrol_python"],
    output_names=["Vout"],
    parameters={
        "R":                 R,
        "L":                 L,
        "C":                 C,
        "Vref_ampl":         VREF_AMPL,
        "freq":              FREQ,
        "usePythonControl":  1.0,   # hand control to Python PI
    },
)
sat     >> rlc_fmu   # saturated control voltage → FMU input
rlc_fmu >> fb_delay  # FMU output (Vout) → delay → error summer (next step)

# 9. Sink – required by VectorSim to identify the end of the signal chain
sink = VectorEnd("sink")
rlc_fmu >> sink

# ---------------------------------------------------------------------------
# Simulation setup and execution
# ---------------------------------------------------------------------------

sim = EmbedSim(sinks=[sink], T=T, dt=dt, solver=ODESolver.EULER)

# Register signals to record: reference and plant output
sim.scope.add(vref,    indices=[0], label="Vref")
sim.scope.add(rlc_fmu, indices=[0], label="Vout")

print("Block diagram (connections):")
sim.print_topology()
print()
sim.print_topology_sources2sink()

n_steps = int(T / dt)
print(f"\nRunning simulation: T={T} s,  dt={dt} s  ({n_steps:,} steps)…")
sim.run()
print(f"Done. Compute time: {sim.stats.compute_time:.3f} s")

# ---------------------------------------------------------------------------
# Post-processing and plot
# ---------------------------------------------------------------------------

t_axis    = np.array(sim.scope.t)          # time vector [s]
Vref_data = sim.scope.get_signal("Vref", index=0)
Vout_data = sim.scope.get_signal("Vout", index=0)

# Tracking error statistics (ignore first half-cycle transient)
steady_mask = t_axis > 1.0 / FREQ          # after first full cycle
e_rms = np.sqrt(np.mean((Vref_data[steady_mask] - Vout_data[steady_mask]) ** 2))
print(f"Steady-state tracking RMS error: {e_rms:.4f} V  "
      f"({100 * e_rms / VREF_AMPL:.2f} % of amplitude)")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t_axis * 1e3, Vref_data, label="Vref (reference)", color="tab:blue",   lw=1.5)
ax.plot(t_axis * 1e3, Vout_data, label="Vout (plant)",     color="tab:orange",  lw=1.5)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Voltage (V)")
ax.set_title("RLC Digital Twin – Reference Tracking (PI Controller)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()

out_path = "rlc_fmu_results.png"
fig.savefig(out_path, dpi=150)
plt.show()
print(f"Plot saved: {out_path}")