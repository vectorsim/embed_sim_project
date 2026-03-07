"""
===============================================================================
RLC DIGITAL TWIN — PI CONTROLLER AUTO-TUNING  (v3 — finite-difference gradient)
===============================================================================

WHAT THIS SCRIPT DOES (plain English)
---------------------------------------
We have an RLC circuit (resistor, inductor, capacitor in series).
We want to control the output voltage Vout to track a sine-wave reference Vref.

A PI controller does this:

    u(t) = Kp * e(t)  +  Ki * integral(e(t))

where  e(t) = Vref(t) − Vout(t)  is the instantaneous tracking error.

The two gains Kp and Ki are unknown.  This script finds good values
automatically by minimising the Integral of Squared Error (ISE):

    J(Kp, Ki)  =  ∫ e(t)²  dt      (over the tuning window)

HOW THE OPTIMISATION WORKS
----------------------------
We use gradient descent with a backtracking (Armijo) line search.

Each iteration:
  1. Evaluate J at the current (Kp, Ki)
  2. Estimate the gradient via central finite differences:
         ∂J/∂Kp  ≈  [ J(Kp+h, Ki) − J(Kp−h, Ki) ] / (2h)
         ∂J/∂Ki  ≈  [ J(Kp, Ki+h) − J(Kp, Ki−h) ] / (2h)
  3. Attempt a normalised gradient step; accept only if ISE improves

WHY FINITE DIFFERENCES INSTEAD OF THE E-L FORMULA
----------------------------------------------------
The Euler-Lagrange impulse-response method would work in an open-loop
setting, but in a *closed-loop* system the PI integral action actively
suppresses any test bump within a few steps, so the estimated impulse
response is identically zero → gradient = 0 → no movement.

Central finite differences directly perturb the objective and are
immune to this problem.  The cost is 4 extra FMU simulations per
iteration (2 per parameter), which is negligible compared to the
accuracy gain.

CIRCUIT PARAMETERS
------------------
    R = 10 Ω,   L = 10 mH,   C = 100 µF
    Reference : 10 V sine at 50 Hz
    Time step : 10 µs
    Tuning window    : 50 ms   (optimisation objective evaluated here)
    Validation window: 150 ms  (final quality check on converged gains)

OUTPUTS
-------
    Optimal Kp, Ki, and integral time Ti = Kp / Ki
    Final RMSE over the validation window
    Four diagnostic plots saved to  rlc_fmu_pi_tuning.png

===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, time

# ── Path setup ────────────────────────────────────────────────────────────────
from _path_utils import get_embedsim_import_path, get_modelica_path
sys.path.insert(0, get_embedsim_import_path())

from embedsim import FMUBlock
from embedsim.core_blocks import VectorSignal


# ===========================================================================
# SIMULATION PARAMETERS
# ===========================================================================

VREF_AMPL = 10.0        # Reference sine-wave amplitude  [V]
FREQ      = 50.0        # Reference frequency            [Hz]
dt        = 1e-5        # Simulation time step           [s]  (10 µs)
T_tune    = 0.05        # Tuning window length           [s]  (50 ms)
T_valid   = 0.15        # Validation window length       [s]  (150 ms)
U_SAT     = 50.0        # Control signal saturation limit [V]

t_tune  = np.arange(0, T_tune,  dt)
t_valid = np.arange(0, T_valid, dt)
N       = len(t_tune)
N_FULL  = len(t_valid)

ref_tune  = VREF_AMPL * np.sin(2 * np.pi * FREQ * t_tune)
ref_valid = VREF_AMPL * np.sin(2 * np.pi * FREQ * t_valid)


# ===========================================================================
# FMU SETUP
# ===========================================================================

FMU_PATH   = get_modelica_path("RLC_Sine_DigitalTwin_OM.fmu")
_FMU_PARAMS = dict(
    fmu_path     = FMU_PATH,
    input_names  = ["Vcontrol_python"],
    output_names = ["Vout"],
    parameters   = {
        "R"               : 10.0,
        "L"               : 10e-3,
        "C"               : 100e-6,
        "Vref_ampl"       : VREF_AMPL,
        "freq"            : FREQ,
        "usePythonControl": 1.0,
    },
)

# Five persistent FMU instances — one per simulation role.
# Re-using instances with reset() is much faster than re-instantiating.
#   plant_base  — baseline cost evaluation
#   plant_kp_p  — Kp + h  perturbation
#   plant_kp_m  — Kp − h  perturbation
#   plant_ki_p  — Ki + h  perturbation
#   plant_ki_m  — Ki − h  perturbation
plant_base = FMUBlock(name="RLC_base", **_FMU_PARAMS)
plant_kp_p = FMUBlock(name="RLC_kp_p", **_FMU_PARAMS)
plant_kp_m = FMUBlock(name="RLC_kp_m", **_FMU_PARAMS)
plant_ki_p = FMUBlock(name="RLC_ki_p", **_FMU_PARAMS)
plant_ki_m = FMUBlock(name="RLC_ki_m", **_FMU_PARAMS)


# ===========================================================================
# CLOSED-LOOP PI SIMULATION — single instance
# ===========================================================================

def run_pi(plant: FMUBlock,
           Kp: float, Ki: float,
           ref: np.ndarray, n: int
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate the closed loop for ``n`` steps using a discrete PI controller.

    The integral term is updated with forward Euler:
        integ  += e_k * dt
        u_k     = clip( Kp * e_k  +  Ki * integ,  −U_SAT, +U_SAT )

    ``plant.reset()`` is called at the start so callers do not need to
    reset manually.

    Returns
    -------
    y     : shape (n,)  — plant output Vout  [V]
    e     : shape (n,)  — tracking error ref − y  [V]
    int_e : shape (n,)  — running integral of e  [V·s]
    """
    y     = np.empty(n)
    e     = np.empty(n)
    int_e = np.empty(n)
    integ  = 0.0
    prev_y = 0.0

    plant.reset()

    for k in range(n):
        e_k      = ref[k] - prev_y
        integ   += e_k * dt
        e[k]     = e_k
        int_e[k] = integ
        u        = np.clip(Kp * e_k + Ki * integ, -U_SAT, U_SAT)
        prev_y   = float(
            plant.compute(k * dt, dt, [VectorSignal([u], "ctrl")]).value[0]
        )
        y[k] = prev_y

    return y, e, int_e


# ===========================================================================
# ISE COST FUNCTION
# ===========================================================================

def ise(plant: FMUBlock, Kp: float, Ki: float) -> float:
    """
    Run one closed-loop simulation and return the ISE cost.

        J = dt * sum( e[k]^2 )   (Riemann-sum approximation of ∫e² dt)

    Parameters
    ----------
    plant : FMUBlock  — which instance to use (keeps evaluations independent)
    Kp, Ki : float    — PI gains to evaluate
    """
    _, e, _ = run_pi(plant, Kp, Ki, ref_tune, N)
    return float(dt * (e @ e))


# ===========================================================================
# GRADIENT VIA CENTRAL FINITE DIFFERENCES
# ===========================================================================

# Perturbation sizes for central differences.
# Rule of thumb: h ≈ sqrt(machine_eps) * |param|  for smooth functions.
# For Kp ~ O(1–10) and Ki ~ O(10–1000) these work well in practice.
H_KP = 0.05     # Kp perturbation step  [dimensionless gain]
H_KI = 0.5      # Ki perturbation step  [A/V·s or 1/s depending on scaling]

def compute_gradient(Kp: float, Ki: float) -> tuple[np.ndarray, float]:
    """
    Estimate the gradient of ISE w.r.t. (Kp, Ki) using central differences,
    and return the baseline ISE.

    Formula
    -------
        ∂J/∂Kp  ≈  [ J(Kp+H_KP, Ki) − J(Kp−H_KP, Ki) ] / (2·H_KP)
        ∂J/∂Ki  ≈  [ J(Kp, Ki+H_KI) − J(Kp, Ki−H_KI) ] / (2·H_KI)

    Five simulations are run (base + 4 perturbed) using independent FMU
    instances so no state bleeds between evaluations.

    Returns
    -------
    grad : np.ndarray, shape (2,)   — [∂J/∂Kp, ∂J/∂Ki]
    ISE  : float                    — baseline cost at (Kp, Ki)
    """
    # Baseline — run first so ISE is always computed at the current point
    J_base = ise(plant_base, Kp, Ki)

    # Kp perturbations — enforce positive gain
    J_kp_p = ise(plant_kp_p, max(0.01, Kp + H_KP), Ki)
    J_kp_m = ise(plant_kp_m, max(0.01, Kp - H_KP), Ki)

    # Ki perturbations — enforce positive gain
    J_ki_p = ise(plant_ki_p, Kp, max(0.01, Ki + H_KI))
    J_ki_m = ise(plant_ki_m, Kp, max(0.01, Ki - H_KI))

    dJ_dKp = (J_kp_p - J_kp_m) / (2.0 * H_KP)
    dJ_dKi = (J_ki_p - J_ki_m) / (2.0 * H_KI)

    return np.array([dJ_dKp, dJ_dKi]), J_base


# ===========================================================================
# OPTIMISATION LOOP  (normalised gradient descent + backtracking)
# ===========================================================================

ALPHA_INIT = 1.0      # Initial step size — large because gradient is normalised
ALPHA_MAX  = 5.0      # Upper bound on step size
ALPHA_GROW = 1.2      # Growth factor when a step is accepted
ALPHA_SHRK = 0.5      # Shrink factor when a step is rejected
MAX_ITER   = 60       # Practical upper bound (converges in ~10–30 iterations)
TOL        = 1e-3     # Stop when ||grad|| < TOL

# Gain bounds
KP_MIN = 0.1
KI_MIN = 1.0

print("=" * 70)
print("  PI Auto-Tuning  —  RLC Digital Twin  (finite-difference gradient)")
print("=" * 70)

Kp, Ki  = 1.0, 10.0    # Conservative starting point — avoids early saturation
alpha   = ALPHA_INIT
history = []            # (Kp, Ki, ISE, |grad|) per iteration
t_start = time.perf_counter()

print(f"\n{'Iter':>5} {'Kp':>10} {'Ki':>10} {'ISE':>14} {'|grad|':>12}  {'s/iter':>8}")
print("-" * 70)

for it in range(MAX_ITER):
    ti = time.perf_counter()

    grad, ISE = compute_gradient(Kp, Ki)
    gnorm     = float(np.linalg.norm(grad))
    history.append((Kp, Ki, ISE, gnorm))

    print(f"{it:>5} {Kp:>10.4f} {Ki:>10.3f} {ISE:>14.6f} {gnorm:>12.6f}  "
          f"{time.perf_counter() - ti:>8.2f}s")

    if gnorm < TOL:
        print("\n  Converged — gradient norm below tolerance.")
        break

    # Normalised step: move exactly `alpha` units in the steepest-descent direction
    direction = grad / (gnorm + 1e-12)
    Kp_new = max(KP_MIN, Kp - alpha * direction[0])
    Ki_new = max(KI_MIN, Ki - alpha * direction[1])

    # Backtracking: only accept the step if ISE strictly decreases.
    # We reuse plant_base here — ise() calls reset() internally.
    ISE_new = ise(plant_base, Kp_new, Ki_new)

    if ISE_new < ISE:
        Kp, Ki = Kp_new, Ki_new
        alpha  = min(alpha * ALPHA_GROW, ALPHA_MAX)   # reward: grow step
    else:
        alpha *= ALPHA_SHRK                            # penalise: shrink step

total_time = time.perf_counter() - t_start
Kp_opt, Ki_opt = Kp, Ki
Ti_opt = Kp_opt / Ki_opt

print(f"\nResult:  Kp = {Kp_opt:.4f},  Ki = {Ki_opt:.4f},  Ti = {Ti_opt:.6f} s")
print(f"Optimisation time: {total_time:.2f} s  ({len(history)} iterations)")


# ===========================================================================
# VALIDATION RUN
# ===========================================================================

plant_final = FMUBlock(name="RLC_final", **_FMU_PARAMS)
y_final, e_final, _ = run_pi(plant_final, Kp_opt, Ki_opt, ref_valid, N_FULL)
rmse = float(np.sqrt(np.mean(e_final ** 2)))
print(f"Final RMSE over {T_valid * 1e3:.0f} ms: {rmse:.5f} V")


# ===========================================================================
# PLOTS
# ===========================================================================

OUTPUT_PNG = "rlc_fmu_pi_tuning.png"
t_ms = t_valid * 1e3

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle("RLC Digital Twin — PI Auto-Tuning", fontsize=14, fontweight="bold")

# Panel 1 — closed-loop tracking
ax = axes[0, 0]
ax.plot(t_ms, ref_valid, lw=1.5, label="Vref", color="tab:blue")
ax.plot(t_ms, y_final,   lw=1.5, label="Vout", color="tab:orange")
ax.set(xlabel="Time (ms)", ylabel="Voltage (V)", title="Closed-Loop Response")
ax.legend()
ax.grid(alpha=0.3)

# Panel 2 — tracking error
ax = axes[0, 1]
ax.plot(t_ms, e_final, lw=1.0, color="tab:red")
ax.set(xlabel="Time (ms)", ylabel="Error (V)",
       title=f"Tracking Error  (RMSE = {rmse:.4f} V)")
ax.grid(alpha=0.3)

# Panel 3 — ISE convergence
ax = axes[1, 0]
ax.semilogy([h[2] for h in history], marker="o", lw=2, color="tab:green")
ax.set(xlabel="Iteration", ylabel="ISE  (log scale)", title="ISE Convergence")
ax.grid(alpha=0.3)

# Panel 4 — gain trajectory
ax  = axes[1, 1]
ax2 = ax.twinx()
ln1, = ax.plot([h[0] for h in history],
               marker="s", color="tab:blue", label="Kp")
ln2, = ax2.plot([h[1] for h in history],
                marker="^", color="tab:purple", linestyle="--", label="Ki")
ax.set(xlabel="Iteration", ylabel="Kp", title="Gain Trajectory")
ax2.set_ylabel("Ki")
ax.legend([ln1, ln2], ["Kp", "Ki"], loc="upper right")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150)
plt.show()
print(f"Plot saved: {OUTPUT_PNG}")
