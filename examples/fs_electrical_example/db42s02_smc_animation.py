#!/usr/bin/env python3
"""
db42s02_smc_sync_animation.py
==============================
Synchronized animation with:
  - 3D (id, iq, RPM) manifold with dynamic path
  - Sliding variable s(t) plot
  - RPM(t) plot
All three plots share the same time cursor and update simultaneously.

FIXED:
  - Manifold now shows full RPM range up to 2000
  - All three plots are properly synchronized
  - Dynamic path shows real-time trajectory
"""

from __future__ import annotations

import sys, os, math, time, contextlib, warnings, logging

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
os.environ["EMBEDSIM_SILENT"] = "1"

import numpy as np
import matplotlib

# 🔥 Smart backend selection (Linux-safe)
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")  # headless mode (no GUI)
else:
    matplotlib.use("TkAgg")  # interactive window

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.stats import norm as sp_norm

from _path_utils import get_project_root, get_embedsim_import_path, get_current_parent

_HERE = get_current_parent()
_ROOT = get_project_root()
_FS_ELEC = _ROOT / "fs_electrical_machines"

for _p in (get_embedsim_import_path(), str(_FS_ELEC), str(_FS_ELEC / "c_src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from embedsim import EmbedSim, ODESolver, VectorEnd
from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE
from embedsim.source_blocks import VectorStep
from embedsim.simulation_engine import VectorDelay

from motor_utility_blocks import SVPWMPackBlock
from svpwm_block import SVPWMBlock
from PMSM_Plant_FMUBlock import PMSM_Plant_FMUBlock
from smc_controller_block import SMCControllerBlock

# Try to import _DB42S02, if not available, define default values
try:
    from smc_controller_block import _DB42S02
except ImportError:
    # Define default DB42S02 parameters if not available
    class _DB42S02:
        SMC_V_DC = 24.0
        SMC_KT = 0.035
        SMC_B_FRICTION = 0.0001
        SMC_I_MAX = 8.0
        SMC_SVPWM_GAIN = 1.0
        SMC_KS_W = 5.55
        SMC_PHI_W = 279.0058
        SMC_KS_I = 0.4992
        SMC_PHI_I = 0.2773
        SMC_LAMBDA_W = 95.3443

_FMU_PATH = str(_FS_ELEC / "modelica" / "PMSM_Plant_FMU.fmu")

# =============================================================================
# Constants
# =============================================================================
V_DC = _DB42S02.SMC_V_DC
TARGET_RPM = 2000.0
TARGET_RADS = TARGET_RPM * 2.0 * math.pi / 60.0
T_SIM = 2.0
DT = 50e-6
T_LOAD_T1 = 0.5
T_LOAD_T2 = 1.2
T_LOAD_ZERO = 0.000
T_LOAD_LIGHT = 0.005
T_LOAD_HEAVY = 0.020
_MOTOR_OUT_SIZE = 8
_IDX_RPM = 0;
_IDX_IA = 1;
_IDX_IB = 2;
_IDX_IC = 3;
_IDX_THETA_M = 4

KT = _DB42S02.SMC_KT
B_FRIC = _DB42S02.SMC_B_FRICTION
I_MAX = _DB42S02.SMC_I_MAX
SVPWM_GAIN = _DB42S02.SMC_SVPWM_GAIN

# Tuner search space
DEFAULTS = np.array([
    _DB42S02.SMC_KS_W,
    _DB42S02.SMC_PHI_W,
    _DB42S02.SMC_KS_I * SVPWM_GAIN,
    _DB42S02.SMC_PHI_I,
    _DB42S02.SMC_LAMBDA_W,
], dtype=np.float64)

PARAM_NAMES = ["KS_W", "PHI_W", "KS_I_raw", "PHI_I", "LAMBDA_W"]
PARAM_UNITS = ["A", "rad/s", "V", "A", "rad/s"]

BOUNDS = np.column_stack([
    np.maximum(DEFAULTS * 0.50, np.array([0.5, 80.0, 0.10, 0.05, 15.0])),
    np.minimum(DEFAULTS * 1.80, np.array([8.0, 1200.0, 3.00, 2.00, 250.0])),
])

W_BUMP = 1.0;
W_SS = 4.0;
W_ID = 25.0;
W_CHAT = 6.0
N_ITER = 20;
PENALTY = 5000.0

# Phase colours with enhanced visibility
S_REACH = 80.0;
C_REACH = "#ff2222"
S_SLIDE = 15.0;
C_SLIDE = "#ffdd00"
C_LOCK = "#22dd22";
C_TRANS = "#2288ff";
ID_TRANS = 0.4

# Animation parameters
TRAIL_LENGTH = 20  # Number of points in trailing line
SLIDE_EMPHASIS_LW = 5.0  # Line width for sliding segments


# =============================================================================
# Silence helper
# =============================================================================
@contextlib.contextmanager
def _silent():
    with open(os.devnull, "w") as dn:
        o, e = sys.stdout, sys.stderr
        sys.stdout = dn;
        sys.stderr = dn
        try:
            yield
        finally:
            sys.stdout = o;
            sys.stderr = e


# =============================================================================
# Plant + CtrlPacker
# =============================================================================
class DB42S02PlantBlock(PMSM_Plant_FMUBlock):
    TOPO_CATEGORY = "plant";
    C_CODEGEN_EXCLUDE = True

    def __init__(self, name, fmu_path):
        super().__init__(name=name, fmu_path=fmu_path)

    def compute_py(self, t, dt, input_values=None):
        ta = tb = tc = 0.5
        if input_values and input_values[0] is not None:
            v = input_values[0].value
            if len(v) >= 3: ta, tb, tc = float(v[0]), float(v[1]), float(v[2])
        # Use load steps for realistic simulation
        if t < T_LOAD_T1:
            tl = T_LOAD_ZERO
        elif t < T_LOAD_T2:
            tl = T_LOAD_LIGHT
        else:
            tl = T_LOAD_HEAVY
        ta = max(0.05, min(0.95, ta));
        tb = max(0.05, min(0.95, tb))
        tc = max(0.05, min(0.95, tc))
        if abs(ta - tb) < 1e-6 and abs(tb - tc) < 1e-6: ta += 1e-4
        return super().compute_py(t, dt, [VectorSignal(
            np.array([ta, tb, tc, V_DC, tl], dtype=DEFAULT_DTYPE))])

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)


class CtrlPacker(VectorBlock):
    C_CODEGEN_EXCLUDE = True

    def __init__(self, name="ctrl_packer", **kw): super().__init__(name, **kw)

    def compute_py(self, t, dt, input_values=None):
        m = (input_values[0].value if input_values and len(input_values) > 0
             else np.zeros(_MOTOR_OUT_SIZE, dtype=DEFAULT_DTYPE))
        r = (input_values[1].value if input_values and len(input_values) > 1
             else np.zeros(1, dtype=DEFAULT_DTYPE))
        self.output = VectorSignal(np.array([
            float(r[0]) if len(r) > 0 else 0.0,
            float(m[_IDX_THETA_M]) if len(m) > _IDX_THETA_M else 0.0,
            float(m[_IDX_IA]) if len(m) > _IDX_IA else 0.0,
            float(m[_IDX_IB]) if len(m) > _IDX_IB else 0.0,
            float(m[_IDX_IC]) if len(m) > _IDX_IC else 0.0,
        ], dtype=DEFAULT_DTYPE), self.name)
        return self.output

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)


# =============================================================================
# Simulation helper
# =============================================================================
def _build_gains(params):
    KS_W, PHI_W, KS_I_raw, PHI_I, LAMBDA_W = params
    return dict(SMC_KS_W=KS_W, SMC_PHI_W=PHI_W,
                SMC_KS_I=KS_I_raw / SVPWM_GAIN,
                SMC_PHI_I=PHI_I, SMC_LAMBDA_W=LAMBDA_W)


def run_sim(gains: dict) -> dict | None:
    try:
        with _silent():
            sr = VectorStep("speed_ref", step_time=0.0,
                            before_value=TARGET_RADS, after_value=TARGET_RADS)
            mot = DB42S02PlantBlock("motor", fmu_path=_FMU_PATH)
            md = VectorDelay("motor_delay", initial=[0.0] * _MOTOR_OUT_SIZE)
            cp = CtrlPacker("ctrl_packer")
            smc = SMCControllerBlock("smc", dt_s=DT, **gains)
            sp = SVPWMPackBlock("svpwm_pack", v_dc=V_DC)
            sp.NUM_INPUTS = 1
            sv = SVPWMBlock("svpwm")
            sk = VectorEnd("sink")
            mot >> md;
            md >> cp;
            sr >> cp;
            cp >> smc;
            smc >> sp;
            sp >> sv;
            sv >> mot;
            mot >> sk
            sim = EmbedSim(sinks=[sk], T=T_SIM, dt=DT, solver=ODESolver.EULER)
            sim.run()
    except Exception as e:
        print(f"Simulation error: {e}")
        return None

    ld = smc.log_data
    if not ld or len(ld.get("t", [])) == 0:
        return None

    t = np.asarray(ld["t"], dtype=np.float64)
    rpm = np.asarray(ld["speed"], dtype=np.float64)
    ref = np.asarray(ld["speed_ref"], dtype=np.float64)
    idd = np.asarray(ld["id"], dtype=np.float64)
    iqq = np.asarray(ld["iq"], dtype=np.float64)
    iqr = np.asarray(ld["iq_ref"], dtype=np.float64)

    lam = gains["SMC_LAMBDA_W"]
    err = (ref - rpm) * 2.0 * math.pi / 60.0
    ie = np.zeros_like(err)
    for k in range(1, len(err)):
        ie[k] = ie[k - 1] + 0.5 * (t[k] - t[k - 1]) * (err[k] + err[k - 1])
    s = err + lam * ie
    return {"t": t, "rpm": rpm, "ref": ref, "id": idd, "iq": iqq, "iq_ref": iqr, "s": s}


# =============================================================================
# Cost from sim dict
# =============================================================================
def cost_metrics(d):
    if d is None: return None
    t = d["t"];
    rpm = d["rpm"];
    idd = d["id"];
    iqr = d["iq_ref"]
    if len(t) < 50 or np.max(np.abs(rpm)) > TARGET_RPM * 2.5: return None

    def bump(ts, win=0.25):
        pre = rpm[t < ts];
        post = rpm[(t >= ts) & (t < ts + win)]
        if len(pre) < 5 or len(post) < 5: return 0.0
        return max(0.0, float(np.mean(pre[-20:])) - float(np.min(post)))

    ss = t > 0.80 * T_SIM
    if not np.any(ss): return None
    br = (bump(T_LOAD_T1) + bump(T_LOAD_T2)) / 2.0
    sse = float(np.mean(np.abs(rpm[ss] - TARGET_RPM)))
    idr = float(np.sqrt(np.mean(idd[ss] ** 2)))
    iqc = float(np.std(iqr[ss]))
    cost = W_BUMP * br + W_SS * sse + W_ID * idr + W_CHAT * iqc
    return {"cost": cost, "bump": br, "ss_error": sse, "id_rms": idr, "iq_chat": iqc}


# =============================================================================
# Latin Hypercube
# =============================================================================
def latin_hypercube(n, bounds, rng, centre=None):
    ndim = len(bounds);
    lo, hi = bounds[:, 0], bounds[:, 1]
    S = np.zeros((n, ndim))
    start = 0
    if centre is not None:
        S[0] = np.clip(centre, lo, hi);
        start = 1
    n_lhs = n - start
    if n_lhs > 0:
        cuts = np.linspace(0, 1, n_lhs + 1)
        for d in range(ndim):
            u = rng.uniform(cuts[:-1], cuts[1:]);
            rng.shuffle(u)
            S[start:, d] = lo[d] + u * (hi[d] - lo[d])
    return np.clip(S, lo, hi)


# =============================================================================
# GP Expected Improvement
# =============================================================================
def next_candidate(X_obs, y_obs, bounds, rng, n_cand=300):
    lo, hi = bounds[:, 0], bounds[:, 1]
    Xn = (X_obs - lo) / (hi - lo + 1e-12)
    kernel = ConstantKernel(1.0, (0.1, 10.0)) * Matern(
        length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                  normalize_y=True, alpha=1e-4)
    gp.fit(Xn, y_obs)
    Xc = rng.uniform(0, 1, size=(n_cand, bounds.shape[0]))
    mu, sig = gp.predict(Xc, return_std=True)
    sig = np.maximum(sig, 1e-9)
    yb = np.min(y_obs);
    Z = (yb - mu - 0.01) / sig
    ei = (yb - mu - 0.01) * sp_norm.cdf(Z) + sig * sp_norm.pdf(Z)
    ei[sig < 1e-8] = 0.0
    return lo + Xc[np.argmax(ei)] * (hi - lo)


# =============================================================================
# RPM manifold surface — physics-derived PMSM steady-state map
# =============================================================================
# Motor parameters from _DB42S02 (or reasonable defaults for DB42S02)
# Stator resistance and inductance for DB42S02
_R_S   = 2.37      # Ω   stator resistance (DB42S02 datasheet)
_L_D   = 3.0e-3    # H   d-axis inductance
_L_Q   = 3.0e-3    # H   q-axis inductance  (surface-mount PMSM: Ld ≈ Lq)
_LAM   = 0.035     # Wb  PM flux linkage  (≈ KT for surface-mount)
_P     = 4         # pole pairs

def build_physics_manifold(id_range, iq_range):
    """
    Steady-state PMSM speed surface in the (id, iq) plane.

    From the q-axis voltage equation at steady state (diq/dt = 0):
        v_q = R·iq + ωe·(λ_pm + Ld·id)

    Bus-limited peak phase voltage: V_ph = V_DC / √3
    (space-vector limit; SVPWM allows up to V_DC/√3)

    Solve for ωe, convert to mechanical RPM:
        ωe   = (V_ph - R·iq) / (λ_pm + Ld·id)   [clipped to ≥ 0]
        RPM  = ωe / P × 60 / (2π)

    id < 0  → flux-weakening (raises ceiling)
    id > 0  → demagnetising  (lowers ceiling)
    iq      → torque axis; higher iq → more volt-drop → lower ceiling
    """
    ID, IQ = np.meshgrid(id_range, iq_range)

    V_ph = V_DC / math.sqrt(3.0)           # ≈ 13.86 V  (SVPWM linear limit)
    lam_eff = _LAM + _L_D * ID             # effective flux linkage
    lam_eff = np.where(lam_eff < 1e-6, 1e-6, lam_eff)

    v_q_available = V_ph - _R_S * IQ       # volt budget on q-axis
    v_q_available = np.maximum(v_q_available, 0.0)

    omega_e = v_q_available / lam_eff      # electrical rad/s
    RPM_grid = omega_e / _P * 60.0 / (2.0 * math.pi)
    RPM_grid = np.clip(RPM_grid, 0.0, TARGET_RPM * 1.35)

    return ID, IQ, RPM_grid


# =============================================================================
# Phase colour per sample
# =============================================================================
def phase_colours(s_arr, id_arr):
    out = []
    for s, id_v in zip(s_arr, id_arr):
        a = abs(s)
        if a > S_REACH:
            out.append(C_REACH)
        elif a > S_SLIDE:
            out.append(C_SLIDE)
        elif abs(id_v) > ID_TRANS:
            out.append(C_TRANS)
        else:
            out.append(C_LOCK)
    return out


# =============================================================================
# SYNCHRONIZED ANIMATION — all three plots share the same time cursor
# =============================================================================
def make_sync_animation(base, tuned, history, best_idx,
                        gif_path="db42s02_smc_sync_animation.gif",
                        png_path="db42s02_smc_sync_animation.png"):
    # Create synchronized time grid for all plots
    # Use the same time base for perfect synchronization
    N = 500  # More frames for smoother animation

    # Interpolate all signals to common time grid
    t_common = np.linspace(tuned["t"][0], tuned["t"][-1], N)

    # Interpolate tuned data
    id_sync = np.interp(t_common, tuned["t"], tuned["id"])
    iq_sync = np.interp(t_common, tuned["t"], tuned["iq"])
    rpm_sync = np.interp(t_common, tuned["t"], tuned["rpm"])
    s_sync = np.interp(t_common, tuned["t"], tuned["s"])
    ref_sync = np.interp(t_common, tuned["t"], tuned["ref"])

    # Interpolate baseline data to common time grid
    base_rpm_sync = np.interp(t_common, base["t"], base["rpm"])
    base_s_sync = np.interp(t_common, base["t"], base["s"])

    # Phase colours for each frame
    cols_all = phase_colours(tuned["s"], tuned["id"])
    idx_map = np.searchsorted(tuned["t"], t_common).clip(0, len(cols_all) - 1)
    col_sync = [cols_all[i] for i in idx_map]

    # Calculate proper axis limits
    actual_max_rpm = max(np.max(tuned["rpm"]), np.max(base["rpm"]))
    target_rpm_max = max(TARGET_RPM * 1.2, actual_max_rpm * 1.05)
    target_rpm_max = min(target_rpm_max, TARGET_RPM * 1.3)

    # Clip for display
    rpm_sync_display = np.clip(rpm_sync, 0, target_rpm_max)
    base_rpm_sync_clipped = np.clip(base_rpm_sync, 0, target_rpm_max)

    # Axis limits
    t0, t1 = t_common[0], t_common[-1]
    s_lim = np.percentile(np.abs(tuned["s"]), 99) * 1.2
    phi_w = _DB42S02.SMC_PHI_W

    id_pad = max(2.0, np.max(np.abs(np.concatenate([tuned["id"], base["id"]]))) * 1.5)
    iq_pad = max(2.5, np.max(np.concatenate([tuned["iq"], base["iq"]])) * 1.3)

    # Statistics
    base_bump = history[0]["bump"]
    best_bump = history[best_idx]["bump"]
    pct = (base_bump - best_bump) / base_bump * 100.0 if base_bump > 0.1 else 0

    # Create figure
    fig = plt.figure(figsize=(26, 11), facecolor="#080808")
    fig.suptitle(
        f"SMC FOC | NANOTEC DB42S02 | 20 kHz | Target: {TARGET_RPM:.0f} RPM | "
        f"Bump: {base_bump:.0f} → {best_bump:.0f} RPM (-{pct:.0f}%)",
        color="white", fontsize=14, fontweight="bold", y=0.98)

    # GridSpec: 1 row, 3 columns
    gs = GridSpec(1, 3, left=0.03, right=0.98, bottom=0.08, top=0.94,
                  wspace=0.25, width_ratios=[1.4, 1.0, 1.0])

    ax3d = fig.add_subplot(gs[0], projection="3d", facecolor="#0c0c0c")
    ax_s = fig.add_subplot(gs[1], facecolor="#0f0f0f")
    ax_sp = fig.add_subplot(gs[2], facecolor="#0f0f0f")

    def _style(ax, xl, yl, title):
        ax.set_facecolor("#0f0f0f")
        ax.tick_params(colors="#cccccc", labelsize=10)
        ax.xaxis.label.set_color("#ffffff")
        ax.yaxis.label.set_color("#ffffff")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333333")
        ax.set_xlabel(xl, fontsize=11)
        ax.set_ylabel(yl, fontsize=11)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=5)
        ax.grid(alpha=0.15, color="#2a2a2a")

    _style(ax_s, "Time [s]", "s  [rad/s]", "Sliding variable s(t)")
    _style(ax_sp, "Time [s]", "Speed [RPM]", f"Velocity (Target: {TARGET_RPM:.0f} RPM)")

    # 3D style
    ax3d.set_facecolor("#0c0c0c")
    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#181818")
    ax3d.tick_params(colors="#cccccc", labelsize=9)
    for lb in (ax3d.xaxis.label, ax3d.yaxis.label, ax3d.zaxis.label):
        lb.set_color("#ffffff")
    ax3d.grid(alpha=0.08)

    # Build physics-derived manifold surface
    # iq axis: allow slightly negative (deceleration transient) up to iq_pad
    iq_min = max(-1.0, -iq_pad * 0.2)
    id_grid = np.linspace(-id_pad, id_pad, 60)
    iq_grid = np.linspace(iq_min, iq_pad, 60)
    ID, IQ, RPM_grid = build_physics_manifold(id_grid, iq_grid)

    # Surface: plasma colourmap, semi-transparent, no edge clutter
    surf = ax3d.plot_surface(ID, IQ, RPM_grid, cmap="plasma", alpha=0.30,
                             edgecolor="none", zorder=1,
                             vmin=0, vmax=TARGET_RPM * 1.1)

    # Target-RPM contour ridge on the surface (where RPM_grid ≈ TARGET_RPM)
    # Draw as a wireframe contour at z = TARGET_RPM for clarity
    ax3d.contour(ID, IQ, RPM_grid, levels=[TARGET_RPM],
                 zdir="z", offset=TARGET_RPM,
                 colors=["#ff4444"], linewidths=2.0, alpha=0.85, zorder=4)

    # Sliding manifold s = 0 ridge:  err + λ·∫err = 0  →  err = 0  →  RPM = TARGET_RPM
    # This is a vertical plane at z = TARGET_RPM (speed-error = 0 hyperplane).
    # Project it as a translucent sheet spanning the (id, iq) axes.
    X_sm = np.linspace(-id_pad, id_pad, 8)
    Y_sm = np.linspace(iq_min, iq_pad, 8)
    Xsm, Ysm = np.meshgrid(X_sm, Y_sm)
    Zsm = np.full_like(Xsm, TARGET_RPM)
    ax3d.plot_surface(Xsm, Ysm, Zsm, color="#22ff88", alpha=0.10,
                      edgecolor="#22ff8840", linewidth=0.4, zorder=2)
    # Label the sliding manifold
    ax3d.text(-id_pad * 0.85, iq_pad * 0.85, TARGET_RPM * 1.02,
              "s = 0", color="#22ff88", fontsize=8, alpha=0.85, zorder=12)

    # Baseline trajectory - use interpolated values
    ax3d.plot(np.clip(np.interp(t_common, base["t"], base["id"]), -id_pad, id_pad),
              np.clip(np.interp(t_common, base["t"], base["iq"]), iq_min, iq_pad),
              base_rpm_sync_clipped,
              color="#888888", lw=1.6, alpha=0.55, zorder=3, label="baseline")

    # Steady-state marker at actual operating point
    _ss = tuned["t"] > tuned["t"][-1] * 0.85
    _ss_id = float(np.mean(tuned["id"][_ss]))
    _ss_iq = float(np.mean(tuned["iq"][_ss]))
    _ss_rpm = float(np.mean(np.clip(tuned["rpm"][_ss], 0, target_rpm_max)))

    ax3d.scatter([_ss_id], [_ss_iq], [_ss_rpm], color="#00ffcc", s=180,
                 marker="*", zorder=10, edgecolors="white", linewidths=1.2,
                 label=f"SS: {_ss_rpm:.0f} RPM")

    # Target marker: solve V_ph = R·iq + ωe·λ  for iq at id=0, ωe=TARGET_RADS*P
    _oe_target = TARGET_RADS * _P
    _V_ph = V_DC / math.sqrt(3.0)
    _iq_target = (_V_ph - _oe_target * _LAM) / _R_S if _V_ph > _oe_target * _LAM else 0.0
    _iq_target = float(np.clip(_iq_target, 0.0, iq_pad))
    ax3d.scatter([0.0], [_iq_target], [TARGET_RPM], color="#ffaa44", s=120,
                 marker="^", zorder=10, edgecolors="white", linewidths=1.0,
                 alpha=0.9, label=f"Target: {TARGET_RPM:.0f} RPM")

    ax3d.set_xlabel("id [A]", fontsize=10, labelpad=5)
    ax3d.set_ylabel("iq [A]", fontsize=10, labelpad=5)
    ax3d.set_zlabel("RPM", fontsize=10, labelpad=5)
    ax3d.set_xlim(-id_pad, id_pad)
    ax3d.set_ylim(iq_min, iq_pad)
    ax3d.set_zlim(0, target_rpm_max)
    ax3d.set_title("(id, iq, RPM) manifold", color="white", fontsize=11, fontweight="bold", pad=6)
    ax3d.view_init(elev=28, azim=-50)
    ax3d.legend(loc="lower left", fontsize=8, labelcolor="white",
                facecolor="#141414", edgecolor="#444444", framealpha=0.85)

    # Static backgrounds for s(t) plot
    ax_s.axhline(0, color="#555555", lw=0.9, ls="--", zorder=1)
    ax_s.axhspan(-phi_w, phi_w, color="#ffaa0010", zorder=0)
    ax_s.axhline(phi_w, color="#ffaa00", lw=1.0, ls=":", alpha=0.55, zorder=2)
    ax_s.axhline(-phi_w, color="#ffaa00", lw=1.0, ls=":", alpha=0.55, zorder=2,
                 label=f"±PHI_W={phi_w:.0f}")

    # Baseline s(t) - use interpolated values
    ax_s.plot(t_common, base_s_sync, color="#44445a", lw=1.2, alpha=0.45, zorder=2,
              label="baseline s")

    # Load change markers
    for ax in (ax_s, ax_sp):
        ax.axvline(T_LOAD_T1, color="orange", ls=":", lw=1.0, alpha=0.4, zorder=1)
        ax.axvline(T_LOAD_T2, color="#ff6666", ls=":", lw=1.0, alpha=0.4, zorder=1)

    ax_s.set_xlim(t0, t1)
    ax_s.set_ylim(-s_lim, s_lim)
    ax_s.legend(loc="upper right", fontsize=8, labelcolor="white",
                facecolor="#141414", edgecolor="#333333", framealpha=0.85)

    # Static backgrounds for RPM plot
    ax_sp.plot(t_common, base_rpm_sync_clipped, color="#777777", lw=1.6, alpha=0.55, zorder=2,
               label="baseline")
    ax_sp.plot(t_common, ref_sync, color="white", lw=1.2, ls="--", alpha=0.5, zorder=2,
               label="ref")

    # Target RPM with tolerance band
    ax_sp.axhline(TARGET_RPM, color="#ff4444", ls="--", lw=1.5, alpha=0.6, zorder=1,
                  label=f"Target: {TARGET_RPM:.0f} RPM")
    rpm_tolerance = TARGET_RPM * 0.01
    ax_sp.axhspan(TARGET_RPM - rpm_tolerance, TARGET_RPM + rpm_tolerance,
                  color="#44ff44", alpha=0.1, zorder=0)

    ax_sp.set_xlim(t0, t1)
    ax_sp.set_ylim(0, target_rpm_max)
    ax_sp.legend(loc="lower right", fontsize=8, labelcolor="white",
                 facecolor="#141414", edgecolor="#333333", framealpha=0.85)

    # ========== ANIMATED ARTISTS ==========

    # 3D: dynamic path + current point
    path_3d, = ax3d.plot([], [], [], lw=2.8, alpha=0.95, solid_capstyle="round", zorder=7)
    live_dot, = ax3d.plot([], [], [], "o", ms=10, zorder=9,
                          markeredgecolor="white", markeredgewidth=0.8)
    drop_ln, = ax3d.plot([], [], [], lw=1.0, ls=":", color="#aaaaaa", alpha=0.6, zorder=3)

    # s(t): dynamic line + current dot
    s_ln, = ax_s.plot([], [], lw=2.2, zorder=5)
    s_dot, = ax_s.plot([], [], "o", ms=8, zorder=8,
                       markeredgecolor="white", markeredgewidth=0.6)

    # RPM: dynamic line + current dot
    sp_ln, = ax_sp.plot([], [], lw=2.5, zorder=4)
    sp_dot, = ax_sp.plot([], [], "o", ms=8, zorder=5,
                         markeredgecolor="white", markeredgewidth=0.6)

    # Time cursors
    cur_s, = ax_s.plot([], [], lw=1.8, color="white", ls="--", alpha=0.7, zorder=6)
    cur_sp, = ax_sp.plot([], [], lw=1.8, color="white", ls="--", alpha=0.7, zorder=6)

    # Phase label
    phase_txt = ax3d.text2D(0.02, 0.97, "", transform=ax3d.transAxes,
                            fontsize=12, fontweight="bold", color=C_LOCK, va="top")
    time_txt = fig.text(0.50, 0.02, "", ha="center", fontsize=11, color="#aaaaaa")

    def update(frame):
        """Update all plots simultaneously"""
        col = col_sync[frame]

        # Get data up to current frame
        ih = id_sync[:frame + 1]
        iqh = iq_sync[:frame + 1]
        rh = rpm_sync_display[:frame + 1]
        th = t_common[:frame + 1]
        sh = s_sync[:frame + 1]

        # Update 3D path
        path_3d.set_data(ih, iqh)
        path_3d.set_3d_properties(rh)
        path_3d.set_color(col)

        # Update 3D dot
        live_dot.set_data([ih[-1]], [iqh[-1]])
        live_dot.set_3d_properties([rh[-1]])
        live_dot.set_color(col)

        # Update drop line (vertical from trajectory point down to RPM=0 floor)
        drop_ln.set_data([ih[-1], ih[-1]], [iqh[-1], iqh[-1]])
        drop_ln.set_3d_properties([rh[-1], 0.0])

        # Update s(t) plot
        s_ln.set_data(th, sh)
        s_ln.set_color(col)
        s_dot.set_data([th[-1]], [sh[-1]])
        s_dot.set_color(col)

        # Update RPM plot
        sp_ln.set_data(th, rh)
        sp_ln.set_color(col)
        sp_dot.set_data([th[-1]], [rh[-1]])
        sp_dot.set_color(col)

        # Update time cursors
        cur_s.set_data([th[-1], th[-1]], [-s_lim, s_lim])
        cur_sp.set_data([th[-1], th[-1]], [0, target_rpm_max])

        # Update labels
        rpm_dev = rh[-1] - TARGET_RPM
        if col == C_LOCK:
            lbl = f"LOCKED  {rh[-1]:.0f} / {TARGET_RPM:.0f} RPM"
        elif col == C_SLIDE:
            lbl = f"SLIDING  s={sh[-1]:.1f}"
        elif col == C_REACH:
            lbl = f"REACHING  s={sh[-1]:.1f}"
        else:
            lbl = "TRANSIENT"
        phase_txt.set_text(lbl)
        phase_txt.set_color(col)

        time_txt.set_text(f"t = {th[-1]:.3f} s  |  RPM = {rh[-1]:.0f}  ({rpm_dev:+3.0f})")

        return (path_3d, live_dot, drop_ln, s_ln, s_dot, sp_ln, sp_dot,
                cur_s, cur_sp, phase_txt, time_txt)

    print(f"\n  Building synchronized animation ({N} frames) ...")
    print(f"  RPM range: 0 → {target_rpm_max:.0f} RPM (target: {TARGET_RPM:.0f})")
    ani = animation.FuncAnimation(fig, update, frames=N, interval=40, blit=False)

    print(f"  Saving {gif_path} ...")
    ani.save(gif_path, writer="pillow", fps=25, dpi=120)
    print(f"  Saved  {gif_path}")

    # Save final frame as PNG
    update(N - 1)
    fig.savefig(png_path, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved  {png_path}")

# =============================================================================
# TUNER (same as original)
# =============================================================================
def run_tuner():
    rng = np.random.default_rng(seed=42)
    print("=" * 65)
    print("  SMC Gain Tuner — Bayesian GP  |  20 iterations")
    print(f"  Baseline: " +
          "  ".join(f"{n}={v:.4f}{u}"
                    for n, u, v in zip(PARAM_NAMES, PARAM_UNITS, DEFAULTS)))
    print("=" * 65)

    history = [];
    X_obs = [];
    y_obs = []
    lhs_pts = latin_hypercube(6, BOUNDS, rng, centre=DEFAULTS)

    for i, params in enumerate(lhs_pts):
        phase = "baseline" if i == 0 else f"LHS-{i}"
        gains = _build_gains(params)
        print(f"\n[{i:2d}/{N_ITER - 1}] {phase}")
        print("         " + "  ".join(f"{n}={v:.4f}{u}"
                                      for n, u, v in zip(PARAM_NAMES, PARAM_UNITS, params)))
        t0 = time.perf_counter()
        d = run_sim(gains);
        m = cost_metrics(d)
        elapsed = time.perf_counter() - t0
        if m is None:
            cost = PENALTY
            m = {"cost": cost, "bump": 999, "ss_error": 999, "id_rms": 99, "iq_chat": 99}
            print(f"         UNSTABLE  ({elapsed:.1f}s)")
        else:
            cost = m["cost"]
            print(f"         cost={cost:.2f}  bump={m['bump']:.1f}RPM"
                  f"  ss={m['ss_error']:.2f}RPM  id={m['id_rms']:.3f}A"
                  f"  ({elapsed:.1f}s)")
        m["params"] = params.copy();
        history.append(m)
        X_obs.append(params);
        y_obs.append(cost)

    for i in range(6, N_ITER):
        params = next_candidate(np.array(X_obs), np.array(y_obs), BOUNDS, rng)
        gains = _build_gains(params)
        print(f"\n[{i:2d}/{N_ITER - 1}] GP-EI")
        print("         " + "  ".join(f"{n}={v:.4f}{u}"
                                      for n, u, v in zip(PARAM_NAMES, PARAM_UNITS, params)))
        t0 = time.perf_counter()
        d = run_sim(gains);
        m = cost_metrics(d)
        elapsed = time.perf_counter() - t0
        if m is None:
            cost = PENALTY
            m = {"cost": cost, "bump": 999, "ss_error": 999, "id_rms": 99, "iq_chat": 99}
            print(f"         UNSTABLE  ({elapsed:.1f}s)")
        else:
            cost = m["cost"]
            print(f"         cost={cost:.2f}  bump={m['bump']:.1f}RPM"
                  f"  ss={m['ss_error']:.2f}RPM  id={m['id_rms']:.3f}A"
                  f"  ({elapsed:.1f}s)")
        m["params"] = params.copy();
        history.append(m)
        X_obs.append(params);
        y_obs.append(cost)

    costs = [h["cost"] for h in history]
    best_idx = int(np.argmin(costs))
    best = history[best_idx]
    best_params = best["params"]
    best_gains = _build_gains(best_params)

    print("\n" + "=" * 65)
    print("  TUNING COMPLETE")
    print("=" * 65)
    print(f"  Baseline (iter  0): cost={history[0]['cost']:.2f}"
          f"  bump={history[0]['bump']:.1f}RPM")
    print(f"  Best     (iter {best_idx:2d}): cost={best['cost']:.2f}"
          f"  bump={best['bump']:.1f}RPM")
    print()
    for name, unit, d, b in zip(PARAM_NAMES, PARAM_UNITS, DEFAULTS, best_params):
        delta = (b - d) / d * 100.0
        print(f"    {name:12s}: {d:.4f} -> {b:.4f} {unit:6s}"
              f"  ({'UP' if b > d else 'DN'} {abs(delta):.1f}%)")
    base_bump = history[0]["bump"];
    best_bump = best["bump"]
    pct = (base_bump - best_bump) / base_bump * 100.0 if base_bump > 0.1 else 0
    print(f"\n  Bump: {base_bump:.1f} -> {best_bump:.1f} RPM  ({pct:.1f}% reduction)")
    print("=" * 65)

    # Write best gains file
    KS_W, PHI_W, KS_I_raw, PHI_I, LAMBDA_W = best_params
    KS_I = KS_I_raw / SVPWM_GAIN
    Path("smc_tuner_best.py").write_text(f"""\
# smc_tuner_best.py — auto-generated
# Best iter {best_idx}  cost={best['cost']:.3f}  bump={best['bump']:.1f}RPM
TUNED_KS_W     = {KS_W:.6f}    # A       (was {_DB42S02.SMC_KS_W:.6f})
TUNED_PHI_W    = {PHI_W:.6f}   # rad/s   (was {_DB42S02.SMC_PHI_W:.6f})
TUNED_KS_I     = {KS_I:.6f}    # V       (was {_DB42S02.SMC_KS_I:.6f})
TUNED_PHI_I    = {PHI_I:.6f}   # A       (was {_DB42S02.SMC_PHI_I:.6f})
TUNED_LAMBDA_W = {LAMBDA_W:.6f}  # rad/s (was {_DB42S02.SMC_LAMBDA_W:.6f})
""")
    print("[Gains] smc_tuner_best.py")
    return history, best_idx, best_gains


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  SMC Tune + Synchronized Animation  —  full pipeline")
    print("=" * 65)

    # 1. Baseline
    print("\n[1/4] Baseline simulation ...")
    base_gains = dict(SMC_KS_W=_DB42S02.SMC_KS_W,
                      SMC_PHI_W=_DB42S02.SMC_PHI_W,
                      SMC_KS_I=_DB42S02.SMC_KS_I,
                      SMC_PHI_I=_DB42S02.SMC_PHI_I,
                      SMC_LAMBDA_W=_DB42S02.SMC_LAMBDA_W)
    base_data = run_sim(base_gains)
    if base_data is None:
        print("  ERROR: Baseline simulation failed!")
        sys.exit(1)
    print(f"  done  final={base_data['rpm'][-1]:.0f} RPM")

    # 2. Tune
    print("\n[2/4] Bayesian GP tuner ...")
    history, best_idx, best_gains = run_tuner()

    # 3. Tuned simulation
    print("\n[3/4] Tuned simulation ...")
    tuned_data = run_sim(best_gains)
    if tuned_data is None:
        print("  ERROR: Tuned simulation failed!")
        sys.exit(1)
    print(f"  done  final={tuned_data['rpm'][-1]:.0f} RPM")

    # 4. Synchronized Animation
    print("\n[4/4] Building synchronized manifold animation ...")
    make_sync_animation(base_data, tuned_data, history, best_idx)

    print("\n[Done]")
    print("  db42s02_smc_sync_animation.gif")
    print("  db42s02_smc_sync_animation.png")
    print("  smc_tuner_best.py")