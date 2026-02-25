"""
===============================================================================
RLC DIGITAL TWIN — PI CONTROLLER AUTO-TUNING
===============================================================================

WHAT THIS SCRIPT DOES (in plain English)
-----------------------------------------
We have an RLC circuit (a resistor, inductor, and capacitor in series).
We want to control the output voltage Vout to track a sine wave reference Vref.

A PI controller does this:
    u(t) = Kp * e(t) + Ki * integral(e(t))

where e(t) = Vref(t) - Vout(t) is the tracking error.

The two gains Kp and Ki are unknown — this script finds good values
automatically by minimising the total squared error (ISE):

    J = integral of e(t)^2 over the tuning window

HOW THE OPTIMISATION WORKS
----------------------------
We use a gradient descent approach. At each iteration:

  1. Run a closed-loop simulation → get error signal e(t)
  2. Estimate the plant impulse response g(t) using a small input bump
  3. Compute how much J would change if we nudged Kp or Ki slightly
     (this is the gradient ∂J/∂Kp and ∂J/∂Ki)
  4. Step the gains in the direction that reduces J

Step 3 uses the Euler-Lagrange sensitivity formula:
    ∂J/∂Kp = - integral( e(t) * (g convolved with e)(t) ) dt
    ∂J/∂Ki = - integral( e(t) * (g convolved with integral_e)(t) ) dt

The convolution is done with FFT for speed (O(N log N) instead of O(N^2)).

PLANT MODEL
-----------
The RLC circuit is packaged as an FMU (Functional Mock-up Unit).
The FMU accepts a control voltage Vcontrol and outputs Vout.
We use two FMU instances:
  - plant_sim : for the main closed-loop pass
  - plant_ir  : for impulse response estimation (runs alongside plant_sim)

PARAMETERS
----------
  R = 10 Ω,  L = 10 mH,  C = 100 µF
  Reference: 10 V sine at 50 Hz
  Time step: 10 µs
  Tuning window: 50 ms  (optimisation runs here)
  Validation window: 150 ms  (final quality check)

OUTPUTS
-------
  - Optimal Kp, Ki, and integral time Ti = Kp/Ki
  - Final RMSE over the validation window
  - Four plots saved to rlc_pi_tuning.png

===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from pathlib import Path
import os, sys, time

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..')))

from embedsim import FMUBlock
from embedsim.core_blocks import VectorSignal

# ── Simulation parameters ─────────────────────────────────────────────────────
VREF_AMPL = 10.0          # Reference amplitude [V]
FREQ      = 50.0          # Reference frequency [Hz]
dt        = 1e-5          # Time step [s]  →  10 µs
T_tune    = 0.05          # Tuning window  [s]  →  50 ms
T_valid   = 0.15          # Validation window [s]  →  150 ms

t_tune  = np.arange(0, T_tune,  dt)
t_valid = np.arange(0, T_valid, dt)
N       = len(t_tune)
N_FULL  = len(t_valid)

ref_tune  = VREF_AMPL * np.sin(2 * np.pi * FREQ * t_tune)
ref_valid = VREF_AMPL * np.sin(2 * np.pi * FREQ * t_valid)

# ── FMU setup ─────────────────────────────────────────────────────────────────
FMU_PATH = Path(_HERE) / "modelica" / "RLC_Sine_DigitalTwin_OM.fmu"
if not FMU_PATH.exists():
    raise FileNotFoundError(f"FMU not found: {FMU_PATH}")

_FMU_PARAMS = dict(
    fmu_path     = FMU_PATH,
    input_names  = ["Vcontrol_python"],
    output_names = ["Vout"],
    parameters   = {
        "R": 10.0, "L": 10e-3, "C": 100e-6,
        "Vref_ampl": VREF_AMPL, "freq": FREQ,
        "usePythonControl": 1.0,
    },
)

# Two persistent instances — avoids repeated FMU instantiation overhead
plant_sim = FMUBlock(name="RLC_sim", **_FMU_PARAMS)
plant_ir  = FMUBlock(name="RLC_ir",  **_FMU_PARAMS)

# ── Closed-loop PI simulation ─────────────────────────────────────────────────
def run_pi(plant, Kp, Ki, ref, n):
    """
    Simulate the closed loop for n steps.
    Returns: output y, error e, integral of error int_e.
    """
    y, e, int_e = np.empty(n), np.empty(n), np.empty(n)
    integ  = 0.0
    prev_y = 0.0
    plant.reset()

    for k in range(n):
        e_k     = ref[k] - prev_y
        integ  += e_k * dt
        e[k]    = e_k
        int_e[k]= integ
        u       = np.clip(Kp * e_k + Ki * integ, -50.0, 50.0)   # saturate at ±50 V
        prev_y  = float(plant.compute(k * dt, dt,
                        [VectorSignal([u], "ctrl")]).value[0])
        y[k]    = prev_y

    return y, e, int_e

# ── Impulse response estimation ───────────────────────────────────────────────
N_IR     = 2000    # Number of samples of impulse response to keep
BUMP_SZ  = 0.5    # Small input bump to estimate plant response [V]
BUMP_IDX = 10     # Step at which the bump is applied

def estimate_impulse_response(Kp, Ki):
    """
    Estimate the plant impulse response g(t) using the bump method:
      1. Run nominal simulation  →  y_nom(t)
      2. Run bumped simulation   →  y_bump(t)  (same gains, +BUMP_SZ at t_bump)
      3. g(t) ≈ (y_bump - y_nom) / BUMP_SZ

    Both simulations run in the same loop to halve FMU calls.
    """
    plant_sim.reset()
    plant_ir.reset()

    y_nom  = np.empty(N)
    y_bump = np.empty(N)
    integ_n = integ_b = prev_n = prev_b = 0.0

    for k in range(N):
        en = ref_tune[k] - prev_n;   integ_n += en * dt
        eb = ref_tune[k] - prev_b;   integ_b += eb * dt
        un = np.clip(Kp * en + Ki * integ_n, -50, 50)
        ub = np.clip(Kp * eb + Ki * integ_b, -50, 50)
        if k == BUMP_IDX:
            ub += BUMP_SZ   # inject the bump on the bumped instance only

        prev_n = float(plant_ir.compute(k * dt, dt,
                       [VectorSignal([un], "n")]).value[0])
        prev_b = float(plant_sim.compute(k * dt, dt,
                       [VectorSignal([ub], "b")]).value[0])
        y_nom[k]  = prev_n
        y_bump[k] = prev_b

    return (y_bump[:N_IR] - y_nom[:N_IR]) / BUMP_SZ

# ── Gradient computation ──────────────────────────────────────────────────────
def compute_gradient(Kp, Ki):
    """
    Compute the E-L gradient of ISE = integral(e^2) w.r.t. Kp and Ki.

    Sensitivity:
        sy_Kp(t) = g(t) convolved with e(t)       [output sensitivity to Kp]
        sy_Ki(t) = g(t) convolved with int_e(t)   [output sensitivity to Ki]

    Gradient:
        dJ/dKp = -integral( e * sy_Kp ) dt
        dJ/dKi = -integral( e * sy_Ki ) dt
    """
    _, e, int_e = run_pi(plant_sim, Kp, Ki, ref_tune, N)
    g_ir  = estimate_impulse_response(Kp, Ki)

    # FFT convolution — causal, truncated to N samples
    sy_Kp = fftconvolve(e,     g_ir, mode='full')[:N] * dt
    sy_Ki = fftconvolve(int_e, g_ir, mode='full')[:N] * dt

    dJ_dKp = -dt * float(e @ sy_Kp)
    dJ_dKi = -dt * float(e @ sy_Ki)
    ISE    =  dt * float(e @ e)

    return np.array([dJ_dKp, dJ_dKi]), ISE

# ── Optimisation loop ─────────────────────────────────────────────────────────
print("=" * 60)
print("  PI Auto-Tuning  —  RLC Digital Twin")
print("=" * 60)

Kp, Ki   = 5.0, 500.0   # Initial guess
alpha    = 0.01          # Gradient step size
max_iter = 25
tol      = 1e-4          # Stop when gradient is this small

history = []
t_start = time.perf_counter()

print(f"\n{'Iter':>5} {'Kp':>10} {'Ki':>10} {'ISE':>14} {'|grad|':>12}  {'s/iter':>8}")
print("-" * 60)

for it in range(max_iter):
    ti = time.perf_counter()
    grad, ISE = compute_gradient(Kp, Ki)
    gnorm = float(np.linalg.norm(grad))
    history.append((Kp, Ki, ISE, gnorm))
    print(f"{it:>5} {Kp:>10.4f} {Ki:>10.2f} {ISE:>14.6f} {gnorm:>12.4f}  "
          f"{time.perf_counter()-ti:>8.2f}s")

    if gnorm < tol:
        print("\n  Converged — gradient is below tolerance.")
        break

    # Normalised gradient step
    step   = alpha * grad / (gnorm + 1e-12)
    Kp_new = max(0.1,  Kp - step[0])
    Ki_new = max(10.0, Ki - step[1])

    # Accept step only if ISE improves (backtracking)
    grad_new, ISE_new = compute_gradient(Kp_new, Ki_new)
    if ISE_new < ISE:
        Kp, Ki = Kp_new, Ki_new
        alpha  = min(alpha * 1.2, 0.1)   # grow step if improving
    else:
        alpha *= 0.5                      # shrink step if not

total_time = time.perf_counter() - t_start
Kp_opt, Ki_opt = Kp, Ki
Ti_opt = Kp_opt / Ki_opt

print(f"\nResult:  Kp = {Kp_opt:.4f},  Ki = {Ki_opt:.4f},  Ti = {Ti_opt:.6f}")
print(f"Optimisation time: {total_time:.2f} s  ({len(history)} iterations)")

# ── Validation run ────────────────────────────────────────────────────────────
plant_final = FMUBlock(name="RLC_final", **_FMU_PARAMS)
y_final, e_final, _ = run_pi(plant_final, Kp_opt, Ki_opt, ref_valid, N_FULL)
rmse = float(np.sqrt(np.mean(e_final**2)))
print(f"Final RMSE over {T_valid*1e3:.0f} ms: {rmse:.5f} V")

# ── Plots ─────────────────────────────────────────────────────────────────────
t_ms  = t_valid * 1e3
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle("RLC Digital Twin — PI Auto-Tuning", fontsize=14, fontweight='bold')

ax = axes[0, 0]
ax.plot(t_ms, ref_valid, lw=1.5, label="Vref",  color="tab:blue")
ax.plot(t_ms, y_final,   lw=1.5, label="Vout",  color="tab:orange")
ax.set(xlabel="Time (ms)", ylabel="Voltage (V)", title="Closed-Loop Response")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(t_ms, e_final, lw=1.0, color="tab:red")
ax.set(xlabel="Time (ms)", ylabel="Error (V)",
       title=f"Tracking Error  (RMSE = {rmse:.4f} V)")
ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.semilogy([h[2] for h in history], marker='o', lw=2, color="tab:green")
ax.set(xlabel="Iteration", ylabel="ISE  (log)", title="ISE Convergence")
ax.grid(alpha=0.3)

ax  = axes[1, 1]
ax2 = ax.twinx()
ln1, = ax.plot( [h[0] for h in history], marker='s', color="tab:blue",   label="Kp")
ln2, = ax2.plot([h[1]/1e3 for h in history], marker='^', color="tab:purple",
                linestyle='--', label="Ki (×10³)")
ax.set(xlabel="Iteration", ylabel="Kp", title="Gain Trajectory")
ax2.set_ylabel("Ki / 1000")
ax.legend([ln1, ln2], ["Kp", "Ki (×10³)"], loc='upper right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("rlc_fmu_pi_tuning.png", dpi=150)
plt.show()
print("Plot saved: rlc_pi_tuning.png")