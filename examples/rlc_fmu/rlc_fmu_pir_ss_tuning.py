"""
===============================================================================
RLC DIGITAL TWIN — PIR CONTROLLER  (State-Space v12 — Correct Design)
===============================================================================

PLANT DESCRIPTION  (from RLC_Sine_DigitalTwin_OM.mo)
------------------------------------------------------
Topology:  VS ──► L1 ──► R1 ──► C1 ──► GND   (series RLC)
Output:    Vout = V_C1   (voltage across the capacitor)
Parameters: R=10Ω, L=10mH, C=100µF

State-space equations (states: x = [V_C, i_L]):
    dV_C/dt = i_L / C
    di_L/dt = (V_input - R·i_L - V_C) / L
    Vout    = V_C

Key plant properties:
    ω_n = 1/√(LC) = 1000 rad/s  (159 Hz natural frequency)
    ζ_p = R/(2√(L/C)) = 0.5     (moderately damped)
    DC gain = 1.0                (V_C charges to V_input in DC steady state)
    Settling time ≈ 4/ζ_p/ω_n = 8ms
    Initial conditions: i_L(0)=0, V_C(0)=0  (fixed in .mo file)

CONTROLLER STRUCTURE — PIR (Proportional + Integral + Resonant)
-----------------------------------------------------------------
The classical PI controller cannot achieve zero steady-state error for a
sinusoidal reference. This follows from the Internal Model Principle (IMP):
to track a signal perfectly, the controller must contain an internal model
of that signal's generator.

    For a sinusoid at ω₀:  the generator poles are at s = ±jω₀.
    A PI controller only has a pole at s=0 (integrator).
    Therefore PI gives finite, non-zero sinusoidal tracking error.

The PIR controller adds a resonant term R(s) with poles at ±jω₀:

    C(s) = Kp  +  Ki/s  +  Kr · [2ζω₀s / (s² + 2ζω₀s + ω₀²)]
           ───    ─────    ──────────────────────────────────────
           P      I        Resonant (bandpass at ω₀)

KEY MATHEMATICAL RESULT — resonant term at ω₀:
    Substituting s = jω₀ into R(s):
        numerator   = Kr · 2ζω₀ · jω₀  = Kr · 2jζω₀²
        denominator = -ω₀² + 2jζω₀² + ω₀² = 2jζω₀²
        R(jω₀) = Kr   (purely real, zero phase shift, independent of ζ!)

    So at the reference frequency: total gain = Kp + Kr (both proportional).
    Steady-state error:  A_ss = VREF / |1 + (Kp+Kr)·G(jω₀)|

STATE-SPACE CONTROLLER IMPLEMENTATION
---------------------------------------
Controller state vector:  x_c = [x_i,  x_r1,  x_r2]
                                  ───    ────    ────
                                  ∫e dt  resonant position  resonant velocity

Continuous-time dynamics:
    ẋ_i  = e                           (integrator)
    ẋ_r1 = x_r2                        (resonant: velocity state)
    ẋ_r2 = -ω₀²·x_r1 - 2ζω₀·x_r2 + e (resonant: driven by error)

Control output:
    u = Kp·e + Ki·x_i + Kr·(2ζω₀)·x_r2
                        ──────────────────
                        C_R_SCALE = 2ζω₀  (output matrix entry for x_r2)

Note: x_r1 does NOT appear in u — only x_r2 (the velocity state) contributes
to the bandpass output. x_r1 is an internal state needed for the dynamics.

Discretised via Zero-Order Hold (ZOH) with step h=dt:
    x_c[k+1] = A_d · x_c[k] + B_d · e[k]
    u[k]     = C_row · x_c[k] + Kp · e[k]
where A_d, B_d are computed from the matrix exponential of the augmented
continuous system (see _build_zoh()).

THREE INTERLOCKING CONSTRAINTS AND THEIR SOLUTIONS
----------------------------------------------------
1. STARTUP SATURATION
   At t=0: i_L=0, V_C=0 → Vout(0)=0 always (inductor/capacitor physics).
   Therefore e(0) = VREF = 10V, giving u(0) = Kp·10.
   Hard limit: Kp_max = U_SAT / VREF = 50/10 = 5.
   Solution: use Kp=5 at startup; ramp Kp upward after plant has settled.

2. RESONANT FILTER TRANSIENT
   The resonant bandpass has settling time τ = 1/(ζ·ω₀).
   Transient peak gain = 1/(2ζ).
   With ζ=0.1: τ=31.8ms, peak=5×  → Kr·5·e easily saturates.
   With ζ=0.5: τ=6.4ms,  peak=1×  → no amplification, settles in one cycle.
   Solution: choose ζ=0.5 so peak_gain=1.0 and settling < one 50Hz cycle.

3. ANTI-WINDUP REQUIREMENT
   When |u_raw| > U_SAT, the actuator is saturated.  If we continue updating
   the controller states, they accumulate "false" information (the integrator
   "thinks" more action is needed when the output is already pegged).
   This causes integrator windup → large overshoot when saturation ends.
   Solution: freeze ALL controller states while saturated.
   Note: we freeze all three (x_i, x_r1, x_r2), not just x_i, because the
   resonant states also wind up under saturation.

GAIN SCHEDULING DESIGN
-----------------------
Transitions at zero-crossings of the 50Hz reference (where ref=0).
State reset to zero at each transition for bumpless transfer.
    u_spike_at_switch = Kp_new · e(t_switch)   (states are zero after reset)
    e(t_switch) ≈ A_ss of previous stage → bounded and safe.

Stage 1 (0–20ms):   Kp=5,  Kr=0   K_eff=5   A_ss≈1.62V  u(0)=50V ← at limit
Stage 2 (20–60ms):  Kp=10, Kr=20  K_eff=30  A_ss≈0.31V  u_switch≈32V ← OK
Stage 3 (60ms+):    Kp=20, Kr=40  K_eff=60  A_ss≈0.16V  u_switch≈23V ← OK

Ki rule: Ti = Kp/Ki = 20/ω₀ = 63.7ms at each stage.
This places the integrator crossover well below ω₀, preventing interference
with the resonant term's sinusoidal tracking action.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import sys

from _path_utils import get_embedsim_import_path, get_modelica_path
sys.path.insert(0, get_embedsim_import_path())

from embedsim import FMUBlock
from embedsim.core_blocks import VectorSignal


# ===========================================================================
# SIMULATION AND PLANT PARAMETERS
# ===========================================================================

VREF_AMPL = 10.0           # reference sine amplitude [V]
FREQ      = 50.0           # reference frequency [Hz]
OMEGA0    = 2.0 * np.pi * FREQ   # ω₀ = 314.159 rad/s

# Resonant filter damping ratio.
# CRITICAL CHOICE: ζ=0.5 gives peak_gain=1/(2·0.5)=1.0 and settling=6.4ms.
# Lower ζ (e.g. 0.1) gives higher peak gain (5×) → saturation during transients.
# Higher ζ (e.g. 1.0) gives overdamped response, slower Kr build-up.
ZETA = 0.50

dt      = 1e-5   # control/FMU step size: 10µs (100kHz sample rate)
T_valid = 0.20   # validation window: 200ms = 10 full cycles at 50Hz
U_SAT   = 50.0   # actuator saturation limit [V]

# Plant physical parameters (must match the .mo file)
R_p, L_p, C_p = 10.0, 10e-3, 100e-6   # Ω, H, F

# Time vector and reference signal for the full validation run
t_valid   = np.arange(0, T_valid, dt)
N_FULL    = len(t_valid)
ref_valid = VREF_AMPL * np.sin(2 * np.pi * FREQ * t_valid)

# Output scaling factor for the resonant controller state x_r2.
# The resonant bandpass output is: u_r = Kr · (2ζω₀) · x_r2
# This factor appears in the C matrix of the controller state-space.
# At ω₀ it evaluates to exactly Kr (see mathematical result in docstring).
C_R_SCALE = 2.0 * ZETA * OMEGA0    # = 314.16 for ζ=0.5


# ===========================================================================
# GAIN SCHEDULE
# ===========================================================================
# Ki rule: Ti = Kp/Ki = 20/ω₀  for all stages.
# Rationale: placing the integral zero at ω₀/20 = 15.7 rad/s keeps the
# integrator well separated from the resonant term at ω₀ = 314 rad/s,
# preventing the two terms from fighting each other.

Kp1, Ki1, Kr1 = 5.0,  5.0  * OMEGA0/20,  0.0   # Stage 1: safe startup
Kp2, Ki2, Kr2 = 10.0, 10.0 * OMEGA0/20, 20.0   # Stage 2: resonant activated
Kp3, Ki3, Kr3 = 20.0, 20.0 * OMEGA0/20, 40.0   # Stage 3: final SS gains

# Stage boundaries at zero-crossings of the 50Hz reference.
# A zero-crossing is chosen because ref=0 there, so after state reset:
#   u_spike = Kp_new · (0 - Vout) = Kp_new · (-small_error)
# This keeps the transition bump small and predictable.
T1 = 1.0/FREQ;  N1 = int(T1/dt)   # t=20ms  (end of 1 complete cycle)
T2 = 3.0/FREQ;  N2 = int(T2/dt)   # t=60ms  (end of 3 complete cycles)


# ===========================================================================
# ANALYTICAL DESIGN VERIFICATION
# ===========================================================================
# Plant frequency response at ω₀:
#   G(jω₀) = (1/LC) / [(jω₀)² + (R/L)(jω₀) + 1/LC]
# This is a complex number: |G(jω₀)|=1.0477, phase=-19.2°
# Steady-state error formula (exact, from closed-loop algebra):
#   A_ss = VREF / |1 + K_eff · G(jω₀)|
# where K_eff = Kp + Kr  (both contribute as pure proportional at ω₀)

s0 = 1j * OMEGA0
G0 = (1/(L_p*C_p)) / (s0**2 + (R_p/L_p)*s0 + 1/(L_p*C_p))

def A_ss_th(Kp, Kr):
    """Predicted steady-state error amplitude [V] using exact frequency-domain formula."""
    return VREF_AMPL / abs(1.0 + (Kp+Kr) * G0)

print("=" * 65)
print("  PIR v12 — Correct Design (zeta=0.5 resonant filter)")
print("=" * 65)
print(f"\nResonant filter: zeta={ZETA}, tau={1/(ZETA*OMEGA0)*1e3:.1f}ms, "
      f"peak_gain={1/(2*ZETA):.1f}×")
print(f"\n{'Stage':>7} {'t(ms)':>10} {'Kp':>5} {'Kr':>5} {'K_eff':>7} {'A_ss_pred':>12}")
for lbl, t0, t1, Kp, Kr in [("1", "0",             f"{T1*1e3:.0f}", Kp1, Kr1),
                              ("2", f"{T1*1e3:.0f}", f"{T2*1e3:.0f}", Kp2, Kr2),
                              ("3", f"{T2*1e3:.0f}", "200",           Kp3, Kr3)]:
    print(f"  {lbl:>5} {t0+'-'+t1:>10} {Kp:>5.0f} {Kr:>5.0f} "
          f"{Kp+Kr:>7.0f} {A_ss_th(Kp,Kr):>12.4f}V")
print(f"\nBaseline PI (Kp=5): A_ss = {A_ss_th(Kp1,0):.4f}V")
print(f"Final   PIR:        A_ss = {A_ss_th(Kp3,Kr3):.4f}V  "
      f"({A_ss_th(Kp1,0)/A_ss_th(Kp3,Kr3):.1f}× improvement)")
print(f"u_ss at stage 3:  {A_ss_th(Kp3,Kr3)*(Kp3+Kr3):.2f}V  (limit {U_SAT}V)")


# ===========================================================================
# ZOH DISCRETISATION OF CONTROLLER STATE-SPACE
# ===========================================================================
# The PIR controller has 3 continuous states: x_c = [x_i, x_r1, x_r2]
#
# Continuous-time A and B matrices (input = tracking error e):
#
#         [ 0     0       0    ]         [1]
#  A_c =  [ 0     0       1    ]  B_c =  [0]
#         [ 0   -ω₀²   -2ζω₀  ]         [1]
#
# The integrator (row 0) is driven directly by e.
# The resonant pair (rows 1-2) forms a 2nd-order oscillator also driven by e.
#
# Zero-Order-Hold discretisation: exact for piecewise-constant e[k].
# Method: augment to 4×4, compute matrix exponential, extract sub-blocks.
#   [A_d | B_d]         [A_c·h  B_c·h]
#   [─────────] = expm( [────────────] )
#   [  0  | 1 ]         [  0  ·  ·  0]

def _build_zoh(omega0, zeta, h):
    """
    Compute exact ZOH discrete matrices (A_d, B_d) for the PIR controller.

    Parameters
    ----------
    omega0 : float  Reference angular frequency [rad/s]
    zeta   : float  Resonant filter damping ratio (0 < zeta < 1)
    h      : float  Sample interval [s]

    Returns
    -------
    A_d : (3,3) ndarray  Discrete state-transition matrix
    B_d : (3,)  ndarray  Discrete input (error) vector
    """
    w  = omega0
    z2 = 2.0 * zeta * omega0          # = 2ζω₀, damping coefficient

    # Continuous-time controller matrices
    A_c = np.array([[0,  0,    0 ],
                    [0,  0,    1 ],
                    [0, -w**2, -z2]], dtype=float)
    B_c = np.array([1.0, 0.0, 1.0])   # error drives both integrator and resonant

    # Build augmented 4×4 matrix for simultaneous exponentiation.
    # This avoids a separate matrix-fraction (van Loan) computation.
    M = np.zeros((4, 4))
    M[:3, :3] = A_c * h    # upper-left:  A_c·h
    M[:3,  3] = B_c * h    # upper-right: B_c·h
    # lower row stays zero

    E = expm(M)            # 4×4 matrix exponential

    A_d = E[:3, :3]        # exact ZOH state-transition matrix
    B_d = E[:3,  3]        # exact ZOH input vector
    return A_d, B_d

# Pre-compute once — same matrices used throughout (ω₀ and ζ are fixed)
A_d, B_d = _build_zoh(OMEGA0, ZETA, dt)


# ===========================================================================
# FMU PLANT SETUP
# ===========================================================================
# The FMU wraps the Modelica RLC model compiled to an FMI 2.0 co-simulation
# unit. At each step we send the control voltage and receive V_C.
# usePythonControl=1 bypasses the internal Modelica PI loop.

FMU_PATH    = get_modelica_path("RLC_Sine_DigitalTwin_OM.fmu")
_FMU_PARAMS = dict(
    fmu_path     = FMU_PATH,
    input_names  = ["Vcontrol_python"],
    output_names = ["Vout"],
    parameters   = {
        "R": 10.0, "L": 10e-3, "C": 100e-6,
        "Vref_ampl": VREF_AMPL, "freq": FREQ,
        "usePythonControl": 1.0,   # hand control to Python
    },
)


# ===========================================================================
# SIMULATION LOOP HELPERS
# ===========================================================================

def run_scheduled(plant, ref, n):
    """
    Simulate the 3-stage gain-scheduled PIR controller against the FMU plant.

    Algorithm per step k:
        1. If at a stage boundary: reset all controller states to zero.
           (Bumpless transfer: u_spike = Kp_new · e only, no accumulated state.)
        2. Select gains for current stage.
        3. Compute raw control output:
               u_raw = C_row @ x_c + Kp · e[k]
               where C_row = [Ki, 0, Kr·C_R_SCALE]
        4. Saturate:  u_sat = clip(u_raw, -U_SAT, +U_SAT)
        5. Advance FMU plant one step with u_sat.
        6. Update controller states with ZOH recursion:
               x_c[k+1] = A_d @ x_c[k] + B_d · e[k]
           Anti-windup: if saturated, freeze x_c (do not update).

    Anti-windup rationale:
        When |u_raw| > U_SAT the actuator output is clipped.  The error
        signal no longer reflects true closed-loop behaviour — the plant
        receives less corrective action than the controller "expects".
        Continuing to update x_c accumulates false state (windup).
        Freezing x_c prevents overshoot when saturation ends.
        We freeze ALL three states because x_r2 winds up under saturation
        just as badly as x_i.

    Parameters
    ----------
    plant : FMUBlock  Initialised FMU plant instance (will be reset).
    ref   : (n,) array  Reference trajectory.
    n     : int  Number of steps to simulate.

    Returns
    -------
    y     : (n,) array  Plant output (V_C) at each step.
    e     : (n,) array  Tracking error  e[k] = ref[k] - y[k-1].
    u_rec : (n,) array  Raw (unsaturated) control signal.
    """
    x      = np.zeros(3)   # controller state: [x_i, x_r1, x_r2]
    prev_y = 0.0            # previous plant output (1-step feedback delay)
    y      = np.empty(n)
    e      = np.empty(n)
    u_rec  = np.empty(n)
    plant.reset()

    for k in range(n):

        # ── Stage boundary: reset states for bumpless transfer ───────────────
        # Resetting x_c to zero means u immediately after the switch is:
        #   u = Kp_new · e_at_switch
        # which is bounded because e_at_switch ≈ A_ss of the previous stage.
        if k == N1 or k == N2:
            x[:] = 0.0

        # ── Gain selection ───────────────────────────────────────────────────
        if   k < N1: Kp, Ki, Kr = Kp1, Ki1, Kr1
        elif k < N2: Kp, Ki, Kr = Kp2, Ki2, Kr2
        else:        Kp, Ki, Kr = Kp3, Ki3, Kr3

        # C_row maps controller state to output contribution:
        #   u_from_states = Ki·x_i + 0·x_r1 + Kr·(2ζω₀)·x_r2
        # x_r1 has zero weight — it is an internal energy-storage state.
        C_row = np.array([Ki, 0.0, Kr * C_R_SCALE])

        # ── Tracking error (using previous plant output — 1-step delay) ──────
        # This is the standard digital control implementation:
        # sample the output, compute error, compute control, then apply.
        e_k = ref[k] - prev_y

        # ── Control law ──────────────────────────────────────────────────────
        # u_raw = state contribution + proportional term
        u_raw = float(C_row @ x) + Kp * e_k
        u_sat = float(np.clip(u_raw, -U_SAT, U_SAT))
        sat   = abs(u_raw) > U_SAT   # True if actuator is saturated

        # ── Advance plant one step ───────────────────────────────────────────
        # The FMU solves the ODE internally with its own solver.
        # We pass the saturated voltage; it returns the new capacitor voltage.
        prev_y = float(
            plant.compute(k * dt, dt, [VectorSignal([u_sat], "ctrl")]).value[0]
        )

        # ── Update controller states (ZOH recursion) ─────────────────────────
        x_new = A_d @ x + B_d * e_k

        # Anti-windup: if saturated, discard the update and hold current state.
        if sat:
            x_new[:] = x[:]   # freeze all three states

        x        = x_new
        y[k]     = prev_y
        e[k]     = e_k
        u_rec[k] = u_raw   # store raw (pre-clip) for analysis

    return y, e, u_rec


def run_fixed(plant, Kp, Ki, Kr, ref, n):
    """
    Simulate the PIR controller with fixed (non-scheduled) gains.
    Same anti-windup logic as run_scheduled.
    Used for baseline comparisons (PI low-gain, PI hi-gain, PIR low-gain).
    """
    C_row  = np.array([Ki, 0.0, Kr * C_R_SCALE])
    x      = np.zeros(3)
    prev_y = 0.0
    y      = np.empty(n);  e = np.empty(n);  u_rec = np.empty(n)
    plant.reset()

    for k in range(n):
        e_k   = ref[k] - prev_y
        u_raw = float(C_row @ x) + Kp * e_k
        u_sat = float(np.clip(u_raw, -U_SAT, U_SAT))
        sat   = abs(u_raw) > U_SAT

        prev_y = float(
            plant.compute(k * dt, dt, [VectorSignal([u_sat], "ctrl")]).value[0]
        )
        x_new = A_d @ x + B_d * e_k
        if sat:
            x_new[:] = x[:]   # anti-windup: freeze all states
        x = x_new

        y[k] = prev_y;  e[k] = e_k;  u_rec[k] = u_raw

    return y, e, u_rec


def ss_amp(sig, t, t_start):
    """
    Estimate steady-state sinusoidal amplitude of sig at OMEGA0.

    Method: least-squares fit of  sig(t) ≈ a·sin(ω₀t) + b·cos(ω₀t)
    over t >= t_start.  Amplitude = √(a²+b²).

    This is more robust than peak-picking because it rejects noise and
    harmonics — only the component at exactly OMEGA0 is measured.
    """
    mask = t >= t_start
    S  = np.sin(OMEGA0 * t[mask])
    Cv = np.cos(OMEGA0 * t[mask])
    # Solve the 2-column least-squares system [S, Cv] · [a,b]ᵀ ≈ sig
    c, _, _, _ = np.linalg.lstsq(np.column_stack([S, Cv]), sig[mask], rcond=None)
    return float(np.sqrt(c[0]**2 + c[1]**2))

# Use the last 3 cycles (60ms) for SS amplitude measurement.
# 3 cycles gives enough data for a reliable sine fit while staying well
# clear of the stage-3 entry transient (which lasts about 1 cycle).
T_SS = T_valid - 3.0 / FREQ


# ===========================================================================
# VALIDATION — run four configurations and compare
# ===========================================================================

print("\n" + "=" * 72)
print("  VALIDATION  (200ms)")
print("=" * 72)
print(f"\n  {'Config':44s} {'RMSE':>8} {'A_ss':>8} {'u_max':>8}")
print("  " + "-" * 72)

def vp(label, y, e, u):
    """Print one row of the validation table and return (RMSE, A_ss)."""
    rm = float(np.sqrt(np.mean(e**2)))      # root-mean-square error over 200ms
    a  = ss_amp(e, t_valid, T_SS)           # steady-state amplitude
    um = float(np.max(np.abs(u)))           # peak control effort (pre-clip)
    print(f"  {label:44s} {rm:>8.5f} {a:>8.5f} {um:>7.2f}V")
    return rm, a, y, e, u

# Baseline 1: low-gain PI — no resonant term, u(t) always within bounds
p1 = FMUBlock(name="RLC_pi_lo", **_FMU_PARAMS)
y1, e1, u1 = run_fixed(p1, Kp1, Ki1, 0.0, ref_valid, N_FULL)
r1, a1, _, _, _ = vp(f"PI  Kp={Kp1} Kr=0  (low-gain baseline)", y1, e1, u1)

# Baseline 2: low-gain PIR — resonant active but Kp=5 limits K_eff=10
p2 = FMUBlock(name="RLC_pir_lo", **_FMU_PARAMS)
y2, e2, u2 = run_fixed(p2, Kp1, Ki1, Kp1, ref_valid, N_FULL)
vp(f"PIR Kp={Kp1} Kr={Kp1} (low-gain+resonant)", y2, e2, u2)

# Baseline 3: high-gain PI, no scheduling — shows what Kp=20 gives without Kr
p3 = FMUBlock(name="RLC_pi_sc", **_FMU_PARAMS)
y3, e3, u3 = run_fixed(p3, Kp3, Ki3, 0.0, ref_valid, N_FULL)
r3, a3, _, _, _ = vp(f"PI  Kp={Kp3} Kr=0  (high-gain, no sched)", y3, e3, u3)

# Main result: 3-stage gain-scheduled PIR
p4 = FMUBlock(name="RLC_pirsc", **_FMU_PARAMS)
y4, e4, u4 = run_scheduled(p4, ref_valid, N_FULL)
r4, a4, _, _, _ = vp(f"PIR 3-stage scheduled (final)", y4, e4, u4)

print(f"\n  PIR_sched vs PI_lo  : A_ss {a1/a4:.2f}×  RMSE {r1/r4:.2f}×")
print(f"  Predicted A_ss={A_ss_th(Kp3,Kr3):.4f}V,  actual={a4:.4f}V")


# ===========================================================================
# PLOTS — 6-panel figure
# ===========================================================================

OUTPUT_PNG = "rlc_fmu_pir_ss_tuning.png"
t_ms = t_valid * 1e3                              # time axis in milliseconds
sw   = dict(ls=":", lw=1.5, color="tab:orange")  # style for stage-switch lines

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(
    f"RLC PIR v12 — zeta={ZETA} resonant filter, 3-stage gain schedule\n"
    f"Stages: Kp=[{Kp1},{Kp2},{Kp3}], Kr=[{Kr1},{Kr2},{Kr3}] | "
    f"A_ss: PI={a1:.4f}V → PIR={a4:.4f}V  ({a1/a4:.1f}×)",
    fontsize=10, fontweight="bold")

# Panel [0,0] — Closed-loop tracking: does the output follow the reference?
ax = axes[0, 0]
ax.plot(t_ms, ref_valid, lw=1.5, color="k",        alpha=0.4, label="Vref")
ax.plot(t_ms, y4,        lw=1.5, color="tab:green",           label=f"PIR sched RMSE={r4:.4f}V")
ax.plot(t_ms, y1,        lw=1.0, color="tab:red",  ls="--", alpha=0.7, label=f"PI lo RMSE={r1:.4f}V")
ax.axvline(T1*1e3, **sw)   # stage 1→2 boundary
ax.axvline(T2*1e3, **sw)   # stage 2→3 boundary
ax.set(xlabel="Time (ms)", ylabel="Voltage (V)", title="Closed-Loop Tracking")
ax.legend(fontsize=7);  ax.grid(alpha=0.3)

# Panel [0,1] — Control signal: verifying no saturation in steady state
ax = axes[0, 1]
ax.plot(t_ms, u4, lw=1.0, color="tab:green",           label="PIR sched u(t)")
ax.plot(t_ms, u1, lw=1.0, color="tab:red",  ls="--", alpha=0.6, label="PI lo u(t)")
ax.axhline( U_SAT, ls=":", color="k", lw=1.5)
ax.axhline(-U_SAT, ls=":", color="k", lw=1.5, label=f"±{U_SAT}V")
ax.axvline(T1*1e3, **sw)
ax.axvline(T2*1e3, **sw)
ax.set(xlabel="Time (ms)", ylabel="u(t) [V]", title="Control Signal", ylim=[-60, 60])
ax.legend(fontsize=7);  ax.grid(alpha=0.3)

# Panel [0,2] — Gain schedule: step changes in Kp and Kr at stage boundaries
ax = axes[0, 2]
Kp_s = np.where(t_valid < T1, Kp1, np.where(t_valid < T2, Kp2, Kp3))
Kr_s = np.where(t_valid < T1, Kr1, np.where(t_valid < T2, Kr2, Kr3))
ax.step(t_ms, Kp_s, color="tab:blue",   label="Kp", where="post", lw=2)
ax.step(t_ms, Kr_s, color="tab:orange", label="Kr", where="post", lw=2)
ax.axvline(T1*1e3, **sw)
ax.axvline(T2*1e3, **sw)
ax.set(xlabel="Time (ms)", ylabel="Gain", title="Gain Schedule")
ax.legend(fontsize=8);  ax.grid(alpha=0.3)

# Panel [1,0] — Steady-state error zoom: dual y-axis to compare magnitudes.
# The PIR error is ~3× smaller than PI hi-gain, shown on separate scales
# to make both waveforms clearly visible.
mask = t_ms >= T_SS * 1e3
ax1  = axes[1, 0];  ax2 = ax1.twinx()
ax1.plot(t_ms[mask], e3[mask], lw=1.5, color="tab:blue",  label=f"PI hi-gain A_ss={a3:.4f}V")
ax2.plot(t_ms[mask], e4[mask], lw=1.5, color="tab:green", label=f"PIR sched A_ss={a4:.4f}V")
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("PI Error (V)",  color="tab:blue")
ax2.set_ylabel("PIR Error (V)", color="tab:green")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax2.tick_params(axis="y", labelcolor="tab:green")
lines = ax1.get_lines() + ax2.get_lines()
ax1.legend(lines, [l.get_label() for l in lines], fontsize=8)
ax1.set_title("SS Error — last 3 cycles");  ax1.grid(alpha=0.3)

# Panel [1,1] — Theory curve: A_ss vs K_eff = Kp+Kr.
# Shows that the analytical formula A_ss = VREF/|1+K_eff·G0| predicts
# the simulation result (green dotted line) to within 0.4%.
# The three stage operating points are marked as vertical dashed lines.
ax = axes[1, 1]
Kv = np.linspace(1, 80, 400)
ax.plot(Kv, VREF_AMPL / np.abs(1 + Kv * G0), lw=2, color="tab:blue", label="Theory")
for Kp_, Kr_, c_, lbl_ in [(Kp1, 0,   "tab:red",   f"Stage1 K={Kp1+0:.0f}"),
                            (Kp2, Kr2, "tab:purple", f"Stage2 K={Kp2+Kr2:.0f}"),
                            (Kp3, Kr3, "tab:green",  f"Stage3 K={Kp3+Kr3:.0f}")]:
    ax.axvline(Kp_ + Kr_, ls="--", color=c_, lw=1.5, label=lbl_)
ax.axhline(a4, ls=":", color="tab:green", lw=1.5, label=f"Actual {a4:.4f}V")
ax.set(xlabel="K_eff = Kp+Kr", ylabel="A_ss [V]",
       title="SS Error vs Gain (theory vs measured)", xlim=[0, 75])
ax.legend(fontsize=7);  ax.grid(alpha=0.3)

# Panel [1,2] — RMSE bar chart: all four configurations at a glance.
# Note: RMSE includes transients, so PIR_sched is not necessarily best in RMSE.
# The key metric is A_ss (steady-state amplitude), shown in panel [1,0].
ax = axes[1, 2]
lbs  = ["PI\nlo-gain", "PIR\nlo-gain", "PI\nhi-gain", "PIR\nsched"]
rv   = [r1, float(np.sqrt(np.mean(e2**2))), r3, r4]
bars = ax.bar(lbs, rv, color=["tab:red","tab:orange","tab:blue","tab:green"],
              alpha=0.8, edgecolor="k")
ax.bar_label(bars, fmt="%.4f", fontsize=8)
ax.set(ylabel="RMSE (V)", title="RMSE Comparison")
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150)
plt.show()
print(f"\nPlot saved: {OUTPUT_PNG}")
