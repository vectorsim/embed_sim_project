"""
===============================================================================
RLC DIGITAL TWIN — GAIN-SCHEDULED LQR WITH INTEGRAL ACTION
===============================================================================

SYSTEM OVERVIEW
---------------
  Series RLC circuit (R=10 Ω, L=10 mH, C=100 µF) driven by a sinusoidal
  reference at 50 Hz. A state-feedback LQR with integral action tracks the
  capacitor voltage V_C. The control input is bounded by ±50 V (actuator
  saturation).

CONTROL ARCHITECTURE
--------------------
  Augmented state vector:  z = [V_C,  i_L,  ξ_I]ᵀ
  Augmented plant:
      ẋ = A_aug·x + B_aug·u           (3rd-order state equation)
      A_aug = | A_p    0  |            A_p = 2×2 RLC state matrix
              | -C_row 0  |            ξ̇_I = -(C·x)  → integral of error
  Control law:
      u = -K·z = -(K_Vc·V_C + K_iL·i_L + K_I·ξ_I)

LQR DESIGN
----------
  Cost functional minimised:
      J = ∫₀^∞ (zᵀQz + uᵀRu) dt
  Q = diag(1/σ_Vc², 1/σ_iL², q_I)      (Bryson's rule normalisation)
  R = R_factor / U_sat²
  Gain K = R⁻¹ Bᵀ P,   P = solution of the algebraic Riccati equation:
      Aᵀ P + P A − P B R⁻¹ Bᵀ P + Q = 0

GAIN SCHEDULING (3 stages)
---------------------------
  Stage 1  t ∈ [0, T₁)      Conservative  K_Vc≈5    q_I=1 000
  Stage 2  t ∈ [T₁, T₂)     Medium        K_Vc≈10   q_I=5 000
  Stage 3  t ∈ [T₂, T_end]  Aggressive    K_Vc≈40   q_I=200 000
  T₁ = 1/f = 20 ms,  T₂ = 3/f = 60 ms
  Bumpless transfer: integrator state ξ_I carried across stage boundaries
  without resetting, preserving continuity of u.

ANTI-WINDUP
-----------
  Back-calculation scheme:
      ξ̇_I ← ξ̇_I + K_t · (u_sat − u_raw)    K_t = 1/K_I
  When u_raw is inside the saturation band, u_sat = u_raw so no correction.
  When saturation is active, ξ_I is driven back to the unsaturated boundary
  at rate proportional to the saturation excess.

STEADY-STATE FREQUENCY ANALYSIS
--------------------------------
  For a sinusoidal reference r(t) = A·sin(ω₀t) the closed-loop steady-state
  phasor of V_C is:
      V_C_ss = C_row · (jω₀I − (A_aug − B_aug·K))⁻¹ · b_ref
  where b_ref encodes how the reference enters the augmented integrator.
  The theoretical tracking gain |V_C_ss|/A and phase lag are computed for
  each stage gain set.

PERFORMANCE METRICS (logged to console)
-----------------------------------------
  • RMSE of tracking error over full horizon
  • Steady-state error amplitude (least-squares sine fit, last 3 cycles)
  • Peak control effort |u|_max vs saturation limit U_sat
  • Closed-loop eigenvalues for each gain stage

SIMULATION NOTES
----------------
  • FMU plant: OpenModelica-exported RLC_Sine_DigitalTwin_OM.fmu
  • Integration: Forward Euler on the inductor current observer; FMU handles
    the internal plant state with its own solver.
  • Timestep dt = 10 µs (100× oversampling of 50 Hz reference)
  • All signals stored at full resolution; plots subsample for readability.

EDUCATIONAL OBJECTIVES
----------------------
  1. State-space modelling of a passive RLC network
  2. Integral action via state augmentation to achieve zero steady-state error
  3. LQR optimal control design via the algebraic Riccati equation
  4. Bryson's rule for practical Q/R weight selection
  5. Gain scheduling: motivating the need and implementing bumpless transfer
  6. Anti-windup by back-calculation for bounded actuators
  7. Resolvent-based closed-loop frequency analysis (no time-domain FFT needed)
  8. Visualising transient vs steady-state behaviour in a servo problem

References
----------
  • Franklin, Powell & Emami-Naeini, "Feedback Control of Dynamic Systems", 8e
  • Åström & Wittenmark, "Computer-Controlled Systems"
  • Skogestad & Postlethwaite, "Multivariable Feedback Design"

Author : Paul (EmbedSim Project)
Date   : 2026-03-06
Version: 2.0  — corrected & annotated plots
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.linalg import solve_continuous_are
import sys

# Custom imports for FMU simulation
from _path_utils import get_embedsim_import_path, get_modelica_path
sys.path.insert(0, get_embedsim_import_path())
from embedsim import FMUBlock
from embedsim.core_blocks import VectorSignal

# ============================================================================ #
# SECTION 1: SYSTEM PARAMETERS
# ============================================================================ #

VREF_AMPL = 10.0           # [V]   Desired reference amplitude
FREQ      = 50.0           # [Hz]  Reference frequency
OMEGA0    = 2.0 * np.pi * FREQ
U_SAT     = 50.0           # [V]   Actuator saturation limit
R_p       = 10.0           # [Ω]   Series resistance
L_p       = 10e-3          # [H]   Inductance
C_p       = 100e-6         # [F]   Capacitance

# Natural frequency and damping ratio (for annotation)
omega_n = 1.0 / np.sqrt(L_p * C_p)       # [rad/s]
zeta    = R_p / (2.0 * np.sqrt(L_p / C_p))

dt       = 1e-5            # [s]   Simulation timestep
T_valid  = 0.20            # [s]   Simulation duration
t_valid  = np.arange(0, T_valid, dt)
N_FULL   = len(t_valid)
ref_valid = VREF_AMPL * np.sin(OMEGA0 * t_valid)

# Stage transition times / indices
T1, T2   = 1.0 / FREQ, 3.0 / FREQ      # 20 ms, 60 ms
N1, N2   = int(T1 / dt), int(T2 / dt)

# ============================================================================ #
# SECTION 2: PLANT STATE-SPACE MATRICES
# ============================================================================ #
#
#  State:  x = [V_C, i_L]ᵀ
#  ẋ = A_p·x + B_p·u,   y = C_p_row·x
#
#  A_p = [ 0         1/C  ]    B_p = [ 0   ]
#        [ -1/(LC)  -R/L  ]          [ 1/L ]
#
A_p     = np.array([[0.0,            1.0 / C_p],
                    [-1.0/(L_p*C_p), -R_p / L_p]])
B_p     = np.array([[0.0], [1.0 / L_p]])
C_p_row = np.array([1.0, 0.0])            # output = V_C

# Augmented system (adds integral state ξ_I):
#   ξ̇_I = -(C·x)  =  r - V_C  (relative to zero reference in reg. form)
A_aug          = np.zeros((3, 3))
A_aug[:2, :2]  = A_p
A_aug[2, :2]   = -C_p_row
B_aug          = np.zeros((3, 1))
B_aug[:2, 0]   = B_p.flatten()

# ============================================================================ #
# SECTION 3: LQR DESIGN FUNCTIONS
# ============================================================================ #

def design_lqr(
    R_factor : float,
    q_I      : float,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Solve the Continuous Algebraic Riccati Equation (CARE) and return LQR
    gains for the 3rd-order augmented RLC system.

    The cost functional minimised is:
        J = ∫₀^∞ (zᵀ Q z + uᵀ R u) dt
    with Q = diag(1/σ_Vc², 1/σ_iL², q_I)  and  R = R_factor / U_sat².
    Bryson's rule normalises state weights by their maximum acceptable
    deviations (σ_Vc = 2 V, σ_iL = 5 A).

    Solution path:
        CARE:  Aᵀ P + P A − P B R⁻¹ Bᵀ P + Q = 0   →  P (3×3 symmetric)
        Gain:  K = R⁻¹ Bᵀ P                          →  K (1×3)

    Parameters
    ----------
    R_factor : float
        Control effort penalty scale factor.
        Smaller value → lower R → less penalty on u → larger gains and
        tighter tracking; larger value → more conservative actuation.
    q_I : float
        LQR weight on the integral state ξ_I.
        Larger value → integrator is penalised less in cost → faster
        integral action and quicker elimination of steady-state error.

    Returns
    -------
    K_vec : np.ndarray, shape (3,)
        LQR gain vector [K_Vc, K_iL, K_I].
        Applied as  u = −K_vec · [V_C, i_L, ξ_I]ᵀ.
    K_I : float
        Integral gain K_vec[2].  Returned separately for convenient
        computation of the anti-windup time constant K_t = 1/K_I.
    eigs : np.ndarray, shape (3,), dtype=complex
        Closed-loop eigenvalues of  (A_aug − B_aug · K_vec).
        All real parts must be strictly negative for stability.
    """
    Q     = np.diag([1.0 / 2.0**2, 1.0 / 5.0**2, q_I])   # Bryson normalisation
    R_lqr = np.array([[R_factor / U_SAT**2]])
    P     = solve_continuous_are(A_aug, B_aug, Q, R_lqr)
    K_mat = (1.0 / R_lqr[0, 0]) * (B_aug.T @ P)
    K_vec = K_mat.flatten()
    eigs  = np.linalg.eigvals(A_aug - B_aug @ K_mat)
    return K_vec, float(K_vec[2]), eigs


def find_R_for_Kvc(
    target_Kvc : float,
    q_I        : float,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Binary-search over R_factor to find the LQR design whose K_Vc matches
    a prescribed target, while holding q_I fixed.

    Motivation: it is more intuitive to specify a desired proportional gain
    on V_C directly (e.g. K_Vc = 40) than to tune the abstract R_factor.
    This wrapper converts that intuitive spec into the correct R_factor via
    60 bisection iterations (relative error < 2⁻⁶⁰ ≈ 10⁻¹⁸).

    Parameters
    ----------
    target_Kvc : float
        Desired value for K_vec[0] (the proportional gain on V_C) [V/V].
    q_I : float
        LQR integral-state weight, passed unchanged to ``design_lqr``.
        Controls the speed of integral action independently of K_Vc.

    Returns
    -------
    K_vec : np.ndarray, shape (3,)
        LQR gain vector [K_Vc, K_iL, K_I] with K_vec[0] ≈ target_Kvc.
    K_I : float
        Integral gain K_vec[2] (anti-windup denominator).
    eigs : np.ndarray, shape (3,), dtype=complex
        Closed-loop eigenvalues of the resulting design.

    Notes
    -----
    Search bounds: R_factor ∈ [1e-8, 10].  K_Vc is a monotonically
    decreasing function of R_factor, so bisection is guaranteed to converge
    provided target_Kvc is achievable within these bounds.
    """
    lo, hi = 1e-8, 10.0
    mid    = (lo + hi) / 2.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        K, _, _ = design_lqr(mid, q_I)
        if K[0] > target_Kvc:
            lo = mid
        else:
            hi = mid
    return design_lqr(mid, q_I)

# ============================================================================ #
# SECTION 4: GAIN SCHEDULE DESIGN
# ============================================================================ #

K1, K_I1, eigs1 = find_R_for_Kvc( 5.0,   1_000)   # Stage 1 — conservative
K2, K_I2, eigs2 = find_R_for_Kvc(10.0,   5_000)   # Stage 2 — medium
K3, K_I3, eigs3 = find_R_for_Kvc(40.0, 200_000)   # Stage 3 — aggressive

Kt1, Kt2, Kt3 = 1.0/K_I1, 1.0/K_I2, 1.0/K_I3    # Anti-windup gains

# Print gain table and eigenvalues
print("\n" + "="*70)
print("  GAIN SCHEDULE SUMMARY")
print("="*70)
for i, (K, eigs, label) in enumerate(
        [(K1, eigs1, "Stage 1 — conservative"),
         (K2, eigs2, "Stage 2 — medium"),
         (K3, eigs3, "Stage 3 — aggressive")], 1):
    print(f"  {label}")
    print(f"    K_Vc={K[0]:8.3f}  K_iL={K[1]:8.3f}  K_I={K[2]:10.1f}")
    for ev in sorted(eigs, key=lambda x: x.real):
        print(f"    eigenvalue: {ev.real:+.1f} {ev.imag:+.1f}j  "
              f"  σ={abs(ev.real):.1f} rad/s  ω_d={abs(ev.imag):.1f} rad/s")
    print()

# ============================================================================ #
# SECTION 5: LQR SIMULATION FUNCTION
# ============================================================================ #

def run_lqr_scheduled(
    plant      : FMUBlock,
    ref        : np.ndarray,
    n          : int,
    full_state : bool = False,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """
    Run the gain-scheduled LQR servo loop with anti-windup and bumpless transfer.

    At each timestep k the loop performs:
      1. Euler integration of the inductor-current observer.
      2. Stage selection based on time index k vs N1, N2 thresholds.
      3. Integral update:  ξ_I += e·dt
      4. State-feedback:   u_raw = −K·[V_C, i_L, ξ_I]ᵀ
      5. Saturation:       u_sat = clip(u_raw, −U_SAT, +U_SAT)
      6. Anti-windup:      ξ_I  += K_t·(u_sat − u_raw)·dt
      7. FMU step:         V_C  ← plant.compute(u_sat)

    Parameters
    ----------
    plant : FMUBlock
        EmbedSim FMU wrapper for the RLC plant.  Must expose a ``reset()``
        method and accept a single ``VectorSignal`` control input.
    ref : np.ndarray, shape (n,)
        Reference voltage trajectory [V].  Typically a 50 Hz sinusoid.
    n : int
        Number of simulation steps to execute.  Must satisfy n ≤ len(ref).
    full_state : bool, optional
        When True the augmented state trajectory z = [V_C, i_L, ξ_I]ᵀ is
        recorded and returned as the 6th element of the output tuple.
        Default: False.

    Returns
    -------
    y : np.ndarray, shape (n,)
        Capacitor voltage V_C measured from the FMU plant [V].
    e : np.ndarray, shape (n,)
        Tracking error e = r − V_C [V].
    u_raw : np.ndarray, shape (n,)
        Control signal before saturation clipping [V].
    u_sat : np.ndarray, shape (n,)
        Control signal actually applied to the plant (after clipping) [V].
    stg : np.ndarray, shape (n,), dtype=int
        Active gain stage index at each step: 1 (conservative),
        2 (medium), or 3 (aggressive).
    z_traj : np.ndarray, shape (n, 3)  — only when full_state=True
        Full augmented state trajectory [V_C, i_L [A], ξ_I [V·s]] at
        every timestep.  Omitted (tuple has 5 elements) when full_state=False.

    Raises
    ------
    IndexError
        If n > len(ref).

    Notes
    -----
    The inductor current i_L is *estimated* by a local Euler observer
    (not read from the FMU), so observer accuracy degrades if dt is large
    relative to L/R = 1 ms.  With dt = 10 µs the error is negligible.

    Anti-windup time constant:  τ_aw = 1 / K_t = K_I  (per stage).
    Bumpless transfer is automatic: ξ_I is never reset at stage boundaries,
    so u is continuous even though K jumps.
    """
    V_C = i_L = xi_I = u_prev = 0.0
    y      = np.empty(n)
    e_arr  = np.empty(n)
    u_raw_ = np.empty(n)
    u_sat_ = np.empty(n)
    stg_   = np.empty(n, int)
    z_traj = np.empty((n, 3)) if full_state else None
    plant.reset()

    for k in range(n):
        # Euler observer for i_L (uses previous actuator output)
        i_L += (u_prev - R_p*i_L - V_C) / L_p * dt

        # Select gain stage
        if   k < N1: K_vec, Kt_aw, stg = K1, Kt1, 1
        elif k < N2: K_vec, Kt_aw, stg = K2, Kt2, 2
        else:        K_vec, Kt_aw, stg = K3, Kt3, 3

        # Error and integral update
        e_k   = ref[k] - V_C
        xi_I += e_k * dt

        # State-feedback control law
        z_c   = np.array([V_C, i_L, xi_I])
        u_r   = -float(K_vec @ z_c)
        u_s   = float(np.clip(u_r, -U_SAT, U_SAT))

        # Anti-windup back-calculation
        xi_I += Kt_aw * (u_s - u_r) * dt
        u_prev = u_s

        # Apply to FMU plant
        result = plant.compute(k*dt, dt, [VectorSignal([u_s], "ctrl")])
        V_C    = float(result.value[0])

        y[k]      = V_C
        e_arr[k]  = e_k
        u_raw_[k] = u_r
        u_sat_[k] = u_s
        stg_[k]   = stg
        if full_state:
            z_traj[k] = z_c

    out = (y, e_arr, u_raw_, u_sat_, stg_)
    return (out + (z_traj,)) if full_state else out

# ============================================================================ #
# SECTION 6: STEADY-STATE HELPERS
# ============================================================================ #

def ss_amp(
    sig     : np.ndarray,
    t       : np.ndarray,
    t_start : float,
) -> float:
    """
    Estimate the steady-state sinusoidal amplitude of a signal by
    least-squares projection onto sin/cos basis functions at OMEGA0.

    The fit solves:
        sig[mask] ≈ c₀·sin(ω₀·t[mask]) + c₁·cos(ω₀·t[mask])
    and returns  A = √(c₀² + c₁²).

    This is equivalent to computing the magnitude of the DFT coefficient
    at exactly ω₀, but avoids windowing artefacts and works for
    non-integer-cycle records.

    Parameters
    ----------
    sig : np.ndarray, shape (N,)
        Signal to analyse (e.g. tracking error e or output V_C) [V].
    t : np.ndarray, shape (N,)
        Time vector corresponding to ``sig`` [s].
    t_start : float
        Start time of the steady-state window [s].  Only samples with
        t ≥ t_start are included in the fit.  Typically set to
        T_valid − 3/FREQ to use the last 3 cycles.

    Returns
    -------
    amplitude : float
        Estimated sinusoidal amplitude [same units as sig].
    """
    mask = t >= t_start
    S    = np.sin(OMEGA0 * t[mask])
    C_   = np.cos(OMEGA0 * t[mask])
    c, _, _, _ = np.linalg.lstsq(np.column_stack([S, C_]), sig[mask], rcond=None)
    return float(np.sqrt(c[0]**2 + c[1]**2))


def cl_phasor(
    A_cl      : np.ndarray,
    B_ref_col : np.ndarray,
    C_row     : np.ndarray,
) -> complex:
    """
    Compute the steady-state output phasor of a closed-loop LTI system
    under a unit sinusoidal input at frequency OMEGA0 (resolvent method).

    For a system  ẋ = A_cl·x + b_ref·r(t),  y = C_row·x  with
    r(t) = e^{jω₀t}, the steady-state response is  y(t) = Y · e^{jω₀t}
    where the output phasor Y satisfies:

        (jω₀I − A_cl) · X = b_ref   →   X = (jω₀I − A_cl)⁻¹ · b_ref
        Y = C_row · X

    The tracking gain is |Y| and the phase lag is −∠Y.

    Parameters
    ----------
    A_cl : np.ndarray, shape (n, n)
        Closed-loop state matrix  A_aug − B_aug · K.
        Must be Hurwitz (all eigenvalues in the open left half-plane).
    B_ref_col : np.ndarray, shape (n,)
        Column vector describing how the sinusoidal reference enters the
        state equation.  For the augmented integrator b_ref = [0, 0, 1]ᵀ.
    C_row : np.ndarray, shape (n,)
        Output row vector.  For V_C: C_row = [1, 0, 0].

    Returns
    -------
    phasor : complex
        Complex output amplitude Y at frequency ω₀.
        |phasor| = steady-state tracking gain (ideal = 1.0 for unity gain).
        angle(phasor) = phase of output relative to input [rad].
    """
    G = np.linalg.solve(1j*OMEGA0*np.eye(A_cl.shape[0]) - A_cl, B_ref_col)
    return complex(C_row @ G)

# ============================================================================ #
# SECTION 7: FMU PLANT SETUP
# ============================================================================ #

FMU_PATH    = get_modelica_path("RLC_Sine_DigitalTwin_OM.fmu")
_FMU_PARAMS = dict(
    fmu_path     = FMU_PATH,
    input_names  = ["Vcontrol_python"],
    output_names = ["Vout"],
    parameters   = {"R": R_p, "L": L_p, "C": C_p,
                    "Vref_ampl": VREF_AMPL, "freq": FREQ,
                    "usePythonControl": 1.0}
)

# ============================================================================ #
# SECTION 8: RUN SIMULATION
# ============================================================================ #

plant = FMUBlock("RLC_lqr", **_FMU_PARAMS)
y, e, u_raw, u_sat, stg, z = run_lqr_scheduled(
    plant, ref_valid, N_FULL, full_state=True)

rmse  = float(np.sqrt(np.mean(e**2)))
a_ss  = ss_amp(e, t_valid, T_valid - 3.0/FREQ)
u_max = float(np.max(np.abs(u_raw)))

print(f"  RMSE (full horizon)      : {rmse:.4f} V")
print(f"  SS error amplitude       : {a_ss:.4f} V   ({100*a_ss/VREF_AMPL:.2f} % of ref)")
print(f"  Peak |u_raw|             : {u_max:.2f} V   (U_sat = {U_SAT:.0f} V)")
print("="*70 + "\n")

# Subsampling factor to keep plots fast (≤ 5 000 points per line)
SS = max(1, N_FULL // 5000)
tv  = t_valid[::SS] * 1e3    # convert to ms for x-axis
t_ms = t_valid * 1e3

# Stage boundary times in ms
T1_ms, T2_ms = T1*1e3, T2*1e3

# ============================================================================ #
# SECTION 9: PLOT
# ============================================================================ #

STAGE_COLORS = ["#d0e8ff", "#ffe0b0", "#d4f5d4"]   # light blue, amber, green
STAGE_LABELS = ["Stage 1\nConservative\n$K_{Vc}=5$",
                "Stage 2\nMedium\n$K_{Vc}=10$",
                "Stage 3\nAggressive\n$K_{Vc}=40$"]

def shade_stages(
    ax   : plt.Axes,
    ymin : float,
    ymax : float,
    alpha: float = 0.18,
) -> None:
    """
    Draw translucent gain-stage background bands and boundary lines on a
    time-domain axes whose x-axis is in milliseconds.

    Three bands are drawn corresponding to the three gain-scheduling stages:
      Stage 1  [0,    T1_ms)   light blue  — conservative gains
      Stage 2  [T1_ms, T2_ms)  amber       — medium gains
      Stage 3  [T2_ms, end]    light green — aggressive gains
    Vertical dashed lines mark the two stage-transition times T1_ms, T2_ms.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes object.  x-axis must already be in units of ms.
    ymin : float
        Lower y-limit hint used internally (not applied to axes limits).
        Kept for API symmetry; axvspan spans the full y range by default.
    ymax : float
        Upper y-limit hint (same note as ymin).
    alpha : float, optional
        Opacity of the stage bands.  Default 0.18 keeps the signal traces
        visible over the background colour.
    """
    boundaries = [(0, T1_ms), (T1_ms, T2_ms), (T2_ms, t_ms[-1]+1)]
    for (x0, x1), col, lbl in zip(boundaries, STAGE_COLORS, STAGE_LABELS):
        ax.axvspan(x0, x1, color=col, alpha=alpha, zorder=0)
    # Stage boundary lines
    for xb in [T1_ms, T2_ms]:
        ax.axvline(xb, color="gray", lw=0.8, ls="--", zorder=1)


fig = plt.figure(figsize=(16, 14))
fig.patch.set_facecolor("#f7f9fc")

# Super-title with key parameters
suptitle_txt = (
    r"$\bf{RLC\ Digital\ Twin}$ — Gain-Scheduled LQR with Integral Action  |  "
    f"R={R_p:.0f} Ω,  L={L_p*1e3:.0f} mH,  C={C_p*1e6:.0f} µF  |  "
    f"$\\omega_n$={omega_n:.0f} rad/s,  ζ={zeta:.2f}  |  "
    f"ref: {VREF_AMPL:.0f} V @ {FREQ:.0f} Hz  |  "
    f"RMSE={rmse:.3f} V,  SS err={a_ss:.3f} V,  |u|_max={u_max:.1f} V"
)
fig.suptitle(suptitle_txt, fontsize=10, y=0.995,
             fontfamily="monospace", color="#1a1a2e")

gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.32,
                      left=0.07, right=0.97, top=0.96, bottom=0.06)

axs = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(2)]
for ax in axs:
    ax.set_facecolor("#fafbfe")

# ── helper to add stage legend ───────────────────────────────────────────────
def add_stage_legend(
    ax : plt.Axes,
) -> None:
    """
    Attach a compact 3-patch legend identifying the gain-stage colour bands
    to the given axes.

    Creates three ``matplotlib.patches.Patch`` handles — one per stage —
    using the module-level STAGE_COLORS palette and labels
    "Stage 1 / 2 / 3".  The legend is placed in the upper-right corner
    with reduced font size and 3 columns to minimise overlap with data.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to attach the legend.  Typically called on
        axs[0] (the output vs reference plot) so the stage key is visible
        at the top of the figure.
    """
    patches = [mpatches.Patch(color=c, alpha=0.5, label=l)
               for c, l in zip(STAGE_COLORS, ["Stage 1", "Stage 2", "Stage 3"])]
    ax.legend(handles=patches, fontsize=6.5, loc="upper right",
              framealpha=0.7, ncol=3)

# ── [0] Output vs Reference ─────────────────────────────────────────────────
ax = axs[0]
shade_stages(ax, -VREF_AMPL*1.3, VREF_AMPL*1.3)
ax.plot(tv, ref_valid[::SS], "--", color="#0077b6", lw=1.2, label="Reference $r$")
ax.plot(tv, y[::SS],          color="#e63946", lw=1.4, label="Output $V_C$")
ax.set_title("① Capacitor Voltage vs Reference", fontweight="bold")
ax.set_xlabel("Time [ms]"); ax.set_ylabel("Voltage [V]")
ax.axhline( VREF_AMPL, color="#adb5bd", lw=0.7, ls=":")
ax.axhline(-VREF_AMPL, color="#adb5bd", lw=0.7, ls=":")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, lw=0.4, color="#cccccc")
# Annotate stage transitions
for xb, lbl in zip([T1_ms, T2_ms], ["Gain\nswitch", "Gain\nswitch"]):
    ax.annotate(lbl, xy=(xb, VREF_AMPL*0.85), fontsize=6.5, color="#555",
                ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-|>", color="#888", lw=0.8),
                xytext=(xb, VREF_AMPL*1.15))

# ── [1] Tracking Error ──────────────────────────────────────────────────────
ax = axs[1]
shade_stages(ax, -12, 12)
ax.plot(tv, e[::SS], color="#f4a261", lw=1.2, label="Error $e = r - V_C$")
ax.axhline(0, color="k", lw=0.6)
ax.fill_between(tv, e[::SS], 0, alpha=0.15, color="#f4a261")
ax.set_title("② Tracking Error $e(t) = r(t) - V_C(t)$", fontweight="bold")
ax.set_xlabel("Time [ms]"); ax.set_ylabel("Error [V]")
ax.legend(fontsize=8)
ax.grid(True, lw=0.4, color="#cccccc")
# Annotate RMSE and SS error
ax.text(0.97, 0.97,
        f"RMSE = {rmse:.3f} V\nSS amp = {a_ss:.3f} V",
        transform=ax.transAxes, fontsize=8, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#aaa", alpha=0.85))

# ── [2] Control Signal ──────────────────────────────────────────────────────
ax = axs[2]
shade_stages(ax, -U_SAT*1.2, U_SAT*1.2)
ax.plot(tv, u_raw[::SS], color="#457b9d", lw=1.3, label="$u_{raw}$ (pre-sat)")
ax.plot(tv, u_sat[::SS], color="#e63946", lw=0.9, ls="--", label="$u_{sat}$ (applied)")
ax.axhline( U_SAT, color="#2d6a4f", lw=1.0, ls="-.",
            label=f"+U_sat = +{U_SAT:.0f} V")
ax.axhline(-U_SAT, color="#2d6a4f", lw=1.0, ls="-.",
            label=f"−U_sat = −{U_SAT:.0f} V")
ax.set_title("③ Control Signal — Pre-saturation vs Applied", fontweight="bold")
ax.set_xlabel("Time [ms]"); ax.set_ylabel("Voltage [V]")
ax.legend(fontsize=7, ncol=2)
ax.grid(True, lw=0.4, color="#cccccc")
ax.text(0.97, 0.97,
        f"|u|_max = {u_max:.1f} V\nU_sat  = {U_SAT:.0f} V",
        transform=ax.transAxes, fontsize=8, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#aaa", alpha=0.85))

# ── [3] Gain Stage with eigenvalue annotation ───────────────────────────────
ax = axs[3]
shade_stages(ax, 0, 4)
ax.step(tv, stg[::SS], where="post", color="#6d6875", lw=2.0)
ax.set_title("④ Active Gain Stage", fontweight="bold")
ax.set_xlabel("Time [ms]"); ax.set_ylabel("Stage index")
ax.set_yticks([1, 2, 3])
ax.set_yticklabels(["1\n(conserv.)", "2\n(medium)", "3\n(aggress.)"])
ax.set_ylim(0.5, 3.8)
ax.grid(True, lw=0.4, color="#cccccc", axis="x")
# Annotate dominant eigenvalue per stage
for stg_idx, eigs_s, xc in [
        (1, eigs1, T1_ms*0.35),
        (2, eigs2, (T1_ms+T2_ms)/2),
        (3, eigs3, (T2_ms + t_ms[-1])*0.55)]:
    dom = sorted(eigs_s, key=lambda v: v.real)[-1]   # least-negative = dominant
    ax.text(xc, stg_idx + 0.25,
            f"λ={dom.real:.0f}{dom.imag:+.0f}j",
            fontsize=6.5, ha="center", color="#333",
            bbox=dict(boxstyle="round,pad=0.2", fc="#ffffffcc", ec="#bbb"))

# ── [4] Integrator State ────────────────────────────────────────────────────
ax = axs[4]
shade_stages(ax, z[:,2].min()*1.1, z[:,2].max()*1.1)
ax.plot(tv, z[::SS, 2], color="#2a9d8f", lw=1.3, label="$\\xi_I$ (integrator state)")
ax.axhline(0, color="k", lw=0.5)
ax.set_title("⑤ Integrator State $\\xi_I$ — Anti-Windup Effect", fontweight="bold")
ax.set_xlabel("Time [ms]"); ax.set_ylabel("$\\xi_I$ [V·s]")
ax.legend(fontsize=8)
ax.grid(True, lw=0.4, color="#cccccc")
ax.text(0.97, 0.97,
        "Anti-windup active when\n$|u_{raw}| > U_{sat}$",
        transform=ax.transAxes, fontsize=7.5, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#aaa", alpha=0.85))

# ── [5] Frequency-Domain Analysis ───────────────────────────────────────────
ax = axs[5]
ax.set_facecolor("#fafbfe")

# Reference entry column in augmented system (sinusoidal ref enters via
# the integral: ξ̇_I += +r(t), so b_ref = [0, 0, 1]ᵀ)
b_ref = np.array([0.0, 0.0, 1.0])

stage_data = [
    ("Stage 1\n(K_Vc=5)",  K1, eigs1, "#74b0d4"),
    ("Stage 2\n(K_Vc=10)", K2, eigs2, "#f4a261"),
    ("Stage 3\n(K_Vc=40)", K3, eigs3, "#2a9d8f"),
]

x_pos    = np.arange(len(stage_data))
gains_db = []
phases   = []
errs_pct = []

for lbl, K_s, _, col in stage_data:
    A_cl   = A_aug - B_aug @ K_s.reshape(1, 3)
    phasor = cl_phasor(A_cl, b_ref, np.array([1.0, 0.0, 0.0]))
    gain   = abs(phasor) / VREF_AMPL          # normalised tracking gain
    phase  = np.degrees(np.angle(phasor))
    err    = 100.0 * abs(1.0 - phasor) / 1.0  # % error vs unity gain ideal
    gains_db.append(20*np.log10(gain + 1e-12))
    phases.append(phase)
    errs_pct.append(100.0 * abs(1.0 - gain))

bar_w = 0.28
bars1 = ax.bar(x_pos - bar_w, gains_db, width=bar_w,
               color=[d[3] for d in stage_data], alpha=0.85,
               label="Tracking gain [dB]")
ax.set_ylabel("Tracking gain [dB]", color="#333")
ax.axhline(0, color="#1a1a2e", lw=0.8, ls="--", label="0 dB (ideal unity)")

ax2 = ax.twinx()
bars2 = ax2.bar(x_pos,        phases,   width=bar_w,
                color=[d[3] for d in stage_data], alpha=0.5, hatch="//",
                label="Phase lag [°]")
bars3 = ax2.bar(x_pos + bar_w, errs_pct, width=bar_w,
                color=[d[3] for d in stage_data], alpha=0.4, hatch="xx",
                label="Gain error [%]")
ax2.set_ylabel("Phase lag [°]  /  Gain error [%]", color="#555")

ax.set_title("⑥ Closed-Loop Frequency Response at 50 Hz\n"
             "(Resolvent analysis — each gain stage)",
             fontweight="bold")
ax.set_xticks(x_pos)
ax.set_xticklabels([d[0] for d in stage_data])
ax.grid(True, lw=0.4, color="#cccccc", axis="y")

# Combined legend
lines = ([mpatches.Patch(fc="gray",   alpha=0.8,  label="Tracking gain [dB]"),
          mpatches.Patch(fc="gray",   alpha=0.5, hatch="//", label="Phase lag [°]"),
          mpatches.Patch(fc="gray",   alpha=0.4, hatch="xx", label="Gain error [%]")])
ax.legend(handles=lines, fontsize=7.5, loc="lower right")

# Annotate numeric values above bars
for i, (gdb, ph, er) in enumerate(zip(gains_db, phases, errs_pct)):
    ax.text(i - bar_w, gdb + 0.05, f"{gdb:.2f}", ha="center",
            va="bottom", fontsize=6.5, color="#111")
    ax2.text(i,        ph + 0.5,   f"{ph:.1f}°",  ha="center",
             va="bottom", fontsize=6.5, color="#444")
    ax2.text(i + bar_w, er + 0.05, f"{er:.2f}%",  ha="center",
             va="bottom", fontsize=6.5, color="#444")

# ── Add stage-legend strip below plot 0 ─────────────────────────────────────
add_stage_legend(axs[0])

# ── Global stage-color legend as figure annotation ──────────────────────────
legend_txt = ("  Stage bands:  "
              "■ Stage 1 (0–20 ms)   "
              "■ Stage 2 (20–60 ms)   "
              "■ Stage 3 (60–200 ms)")
fig.text(0.5, 0.005, legend_txt, ha="center", fontsize=8,
         color="#444", style="italic")

plt.savefig("rlc_lqr_servo_plots.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print("Plot saved.")
