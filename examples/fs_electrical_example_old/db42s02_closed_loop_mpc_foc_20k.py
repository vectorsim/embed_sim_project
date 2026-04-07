# db42s02_closed_loop_mpc_foc_20k.py
"""
db42s02_closed_loop_mpc_foc_20k.py
===================================
EmbedSim  —  Closed-loop MPC FOC  —  NANOTEC DB42S02  —  AURIX TC3xx test
==========================================================================

Architecture (encoder-based):
  theta_e  = p·theta_m              exact from encoder → Park / InvPark
  omega_m  = Δtheta_m/dt + IIR      encoder speed      → speed reference
  SMO      = ê_α_filt, ê_β_filt     back-EMF filter    → disturbance feedforward
  MPC      = predicts and optimizes vd, vq over horizon N → optimal control
  InvPark → SVPWM → ta,tb,tc → AURIX GTM

MPC solves at each step:
  minimize: Σ (x_ref - x)ᵀQ(x_ref - x) + uᵀR u
  subject to: x(k+1) = A·x(k) + B·u(k) + f(k)  (with SMO disturbance)
              |vd| ≤ V_MAX, |vq| ≤ V_MAX
              |id| ≤ I_MAX, |iq| ≤ I_MAX

Load schedule (simulation only):
  t < 0.5s  : no load
  0.5–1.2s  : 5 mN·m
  1.2–2.0s  : 20 mN·m

CodeGen  →  embedsim_gen/embedsim_step.c / .h
"""

from __future__ import annotations

import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from _path_utils import get_project_root, get_embedsim_import_path, get_current_parent

_HERE    = get_current_parent()
_ROOT    = get_project_root()
_FS_ELEC = _ROOT / "fs_electrical_machines"

for _p in (get_embedsim_import_path(), str(_FS_ELEC), str(_FS_ELEC / "c_src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from embedsim import EmbedSim, ODESolver, VectorEnd
from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE
from embedsim.source_blocks import VectorStep
from embedsim.simulation_engine import VectorDelay
from embedsim.code_generator import CodeGenStart, CodeGenEnd

from motor_utility_blocks import SVPWMPackBlock
from svpwm_block import SVPWMBlock
from mpc_controller_block import MPCControllerBlock, _DB42S02 as MPC_MOTOR
from PMSM_Plant_FMUBlock import PMSM_Plant_FMUBlock


# =============================================================================
# Test parameters — change these for different operating points
# =============================================================================

V_DC       = 17.0                # V
TARGET_RPM = 2000.0              # [RPM]  target speed
T_SIM      = 5.0                 # [s]    simulation duration
DT         = 50e-6               # [s]    20 kHz — matches AURIX GTM period
_RAMP_TIME = 0.5                 # [s]    linear speed ramp duration

# Load schedule (simulation only — not present on hardware)
T_LOAD_T1    = 0.5    # [s]  light load starts
T_LOAD_T2    = 1.2    # [s]  heavy load starts
T_LOAD_ZERO  = 0.000  # [N·m]
T_LOAD_LIGHT = 0.005  # [N·m]  5 mN·m
T_LOAD_HEAVY = 0.020  # [N·m]  20 mN·m

TARGET_RADS_MECH = TARGET_RPM * 2.0 * math.pi / 60.0
_MOTOR_OUT_SIZE  = 8   # FMU: [rpm(0), ia(1), ib(2), ic(3), theta_m(4),
                        #       T_em(5), id_out(6), iq_out(7)]

# =============================================================================
# Sensor noise configuration
# =============================================================================
# Set ENABLE_NOISE = True to activate Phase 2 / Phase 3 validation.
# All noise values are 1-sigma (standard deviation).
#
# ADC current noise:
#   12-bit ADC on AURIX, ±I_MAX range → LSB = 2×3.57/4096 ≈ 1.74 mA
#   Switching noise adds ~5–10 mA RMS on top.
#   NOISE_I_SIGMA = 0.01 A is a conservative but realistic 1-sigma.
#
# Encoder quantisation:
#   1000 PPR encoder → resolution = 2π/4000 ≈ 1.57 mrad per count.
#   Model as Gaussian with sigma = 0.5 LSB = 0.78 mrad.
#   NOISE_THETA_SIGMA = 0.001 rad (slightly above quantisation floor,
#   accounts for bearing slop, cable flex, interpolation error).
#
# These are independent, zero-mean, white noise sources.
# The SMO low-pass filter (fc=1000 Hz) attenuates current noise;
# the boundary layer phi_i handles residual chattering.

ENABLE_NOISE      = False    # ← set True to activate noise injection

NOISE_I_SIGMA     = 0.01    # [A]    1-sigma ADC current noise  (ia, ib, ic)
NOISE_THETA_SIGMA = 0.001   # [rad]  1-sigma encoder angle noise (theta_m)
NOISE_SEED        = 42      # reproducible runs; set None for random

# =============================================================================
# MPC parameters — intuitive weights, no gain tuning!
# =============================================================================
#
# MPC PRINCIPLES:
#   Prediction horizon N = 10 (500µs at 20 kHz)
#   Q matrix: state tracking weights (higher = more important)
#     Q_id    = 10.0   — keep id near zero (MTPA)
#     Q_iq    = 0.1    — iq regularisation: appears in vq denominator only,
#                        damping the vq gain without creating a competing
#                        numerator term. Must be << Q_omega for speed control.
#     Q_omega = 500.0  — speed tracking (dominant weight, drives vq numerator)
#   R matrix: control effort weights (higher = less aggressive)
#     R_vd = 0.01  — allow vd to change freely
#     R_vq = 0.01  — allow vq to change freely

MPC_N        = 10       # Prediction horizon (steps)
MPC_Q_ID     = 10.0     # id tracking weight  (drive id → 0, MTPA)
MPC_Q_IQ     = 0.1     # iq regularisation weight (denominator only — damps vq gain)
MPC_Q_OMEGA  = 500.0   # speed tracking weight (must dominate Q_iq for speed control)
MPC_R_VD     = 0.01    # vd control effort weight
MPC_R_VQ     = 0.01    # vq control effort weight

# SMO parameters (for disturbance rejection)
MPC_SMO_K    = 4.68     # V (4× max back-EMF for robust convergence)
MPC_SMO_FC   = 1000.0   # Hz (LPF cutoff for smooth back-EMF)


# =============================================================================
# Plant block — PMSM_Plant_FMUBlock  (PMSM_Plant_FMU.fmu)
# =============================================================================
# FMU input bus : [duty_a, duty_b, duty_c, v_dc, T_load]
# FMU output bus: [rpm(0), ia(1), ib(2), ic(3), theta_m(4),
#                  T_em(5), id_out(6), iq_out(7)]
#
# compute_py pattern from db42s02_openloop_fmu.py:
#   - duties read unconditionally (no zero-guard)
#   - FMU input bus packed as VectorSignal and passed to super().compute_py
#   - T_load injected from timed schedule (never in CodeGen region)

_FMU_PATH = str(_FS_ELEC / "modelica" / "PMSM_Plant_FMU.fmu")


class DB42S02PlantBlock(PMSM_Plant_FMUBlock):
    """
    DB42S02 FMU plant with timed load torque schedule.
    Wraps PMSM_Plant_FMUBlock with the 5-element input bus
    [duty_a, duty_b, duty_c, v_dc, T_load].
    """

    TOPO_CATEGORY     = "plant"
    C_CODEGEN_EXCLUDE = True
    output_label      = "[rpm,ia,ib,ic,theta_m,Tem,id,iq]"

    def __init__(self, name: str):
        super().__init__(name=name, fmu_path=_FMU_PATH)

    def compute_py(self, t, dt, input_values=None):
        if   t < T_LOAD_T1: t_load = T_LOAD_ZERO
        elif t < T_LOAD_T2: t_load = T_LOAD_LIGHT
        else:                t_load = T_LOAD_HEAVY

        # Duties read unconditionally — zero-vector states are legitimate
        ta = tb = tc = 0.5
        if input_values and input_values[0] is not None:
            v = input_values[0].value
            if len(v) >= 3:
                ta, tb, tc = float(v[0]), float(v[1]), float(v[2])

        # Pack FMU input bus: [duty_a, duty_b, duty_c, v_dc, T_load]
        fmu_in = VectorSignal(
            np.array([ta, tb, tc, V_DC, t_load], dtype=DEFAULT_DTYPE))
        return super().compute_py(t, dt, [fmu_in])

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)


# =============================================================================
# CtrlPacker — feedback packer + CodeGen input naming
# =============================================================================

class CtrlPacker(VectorBlock):
    """
    Packs plant feedback into the MPC input bus.
    Defines named input fields for CodeGen (INPUT_NAMES → EmbedSim_Input_T).

    Outputs: [omega_ref_mech, theta_m, ia, ib, ic, omega_m_meas]
      omega_ref_mech — rate-limited ramp [rad/s]
      theta_m        — unwrapped mechanical angle [rad]
      ia, ib, ic     — phase currents [A]
      omega_m_meas   — mechanical speed from FMU rpm output [rad/s]

    Note on theta_m unwrapping
    --------------------------
    The FMU integrates theta_e and outputs theta_m = theta_e / p, which wraps
    every 2π/p mechanical radians (= π/2 rad for p=4).  The MPC Park transform
    computes theta_e = p * theta_m — this must receive a continuously increasing
    angle, not one that resets to 0 every π/2 rad.  A 2π unwrap is applied here
    on the raw FMU theta_m before forwarding it to the controller, mirroring the
    _last_theta_m_unwrapped pattern in smc_controller_block.py.
    """

    INPUT_NAMES       = ["omega_ref_mech", "theta_m", "ia", "ib", "ic"]
    INPUT_KEEP        = [0, 1, 2, 3, 4]
    C_CODEGEN_EXCLUDE = True

    _RAMP_RATE = TARGET_RADS_MECH / _RAMP_TIME   # rad/s²

    def __init__(self, name: str = "ctrl_packer", **kw):
        super().__init__(name, **kw)
        self.output_label            = "[ω_ref,θ_m,ia,ib,ic,ω_m]"
        self._omega_ref_filt: float  = 0.0
        self._theta_m_prev: float    = 0.0      # previous raw FMU theta_m
        self._theta_m_unwrapped: float = 0.0    # continuously accumulating angle
        self._rng = np.random.default_rng(NOISE_SEED)

    def reset(self):
        super().reset()
        self._omega_ref_filt    = 0.0
        self._theta_m_prev      = 0.0
        self._theta_m_unwrapped = 0.0
        self._rng = np.random.default_rng(NOISE_SEED)

    def compute_py(self, t, dt, input_values=None):
        m = (input_values[0].value
             if input_values and len(input_values) > 0
             else np.zeros(_MOTOR_OUT_SIZE, dtype=DEFAULT_DTYPE))
        r = (input_values[1].value
             if input_values and len(input_values) > 1
             else np.zeros(1, dtype=DEFAULT_DTYPE))

        # Linear speed ramp
        omega_target = float(r[0]) if len(r) > 0 else 0.0
        max_step = self._RAMP_RATE * dt
        self._omega_ref_filt += max(
            -max_step,
            min(max_step, omega_target - self._omega_ref_filt))

        # FMU output bus: [rpm(0), ia(1), ib(2), ic(3), theta_m(4), T_em(5), ...]
        # FMU theta_m = theta_e / p — wraps every 2π/p rad (π/2 rad for p=4).
        # Unwrap to continuously accumulating angle for MPC Park transform.
        theta_m_raw = float(m[4]) if len(m) > 4 else 0.0
        delta = theta_m_raw - self._theta_m_prev
        delta -= 2.0 * math.pi * math.floor(
            (delta + math.pi) / (2.0 * math.pi))
        self._theta_m_unwrapped += delta
        self._theta_m_prev       = theta_m_raw
        theta_m = self._theta_m_unwrapped

        ia = float(m[1]) if len(m) > 1 else 0.0
        ib = float(m[2]) if len(m) > 2 else 0.0
        ic = float(m[3]) if len(m) > 3 else 0.0

        # Exact mechanical speed from FMU rpm[0] — zero lag.
        rpm_val      = float(m[0]) if len(m) > 0 else 0.0
        omega_m_meas = rpm_val * (2.0 * math.pi / 60.0)

        # ── Sensor noise injection ────────────────────────────────────────────
        if ENABLE_NOISE:
            ia      += float(self._rng.normal(0.0, NOISE_I_SIGMA))
            ib      += float(self._rng.normal(0.0, NOISE_I_SIGMA))
            ic      += float(self._rng.normal(0.0, NOISE_I_SIGMA))
            theta_m += float(self._rng.normal(0.0, NOISE_THETA_SIGMA))

        self.output = VectorSignal(np.array([
            self._omega_ref_filt,
            theta_m,
            ia,
            ib,
            ic,
            omega_m_meas,
        ], dtype=DEFAULT_DTYPE), self.name)
        return self.output

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)


# =============================================================================
# Build, simulate, generate code
# =============================================================================
# CodeGen-configured SVPWM subclasses
# =============================================================================
# C_CUSTOM_EMIT (and the other CodeGen class attributes) are declared as
# read-only @property on the base classes.  They must be overridden at the
# class level via subclassing — instance-level assignment raises AttributeError.
# Pattern mirrors SMCControllerBlock.C_CUSTOM_EMIT in smc_controller_block.py.

class DB42S02SVPWMPackBlock(SVPWMPackBlock):
    """SVPWMPackBlock with DB42S02 / AURIX CodeGen metadata."""

    C_SOURCES    = ["embed_sim_motor_utility_blocks.c"]
    C_HEADERS    = ["embed_sim_motor_utility_blocks.h"]
    state_struct = "SVPWMPack_T"
    step_func    = "SVPWMPack_Step"
    init_func    = "SVPWMPack_Init"
    C_INIT_ARGS  = ["v_dc"]
    C_CUSTOM_EMIT = (
        "    /* --- svpwm_pack (SVPWMPackBlock) --- */\n"
        "    {\n"
        "        SVPWMPack_T  svpwm_pack_st;\n"
        "        real32_T     y_svpwm_pack[3];\n"
        "        real32_T     u_svpwm_pack[2];\n"
        "        u_svpwm_pack[0] = y_mpc[0];   /* v_alpha */\n"
        "        u_svpwm_pack[1] = y_mpc[1];   /* v_beta  */\n"
        "        SVPWMPack_Init(&svpwm_pack_st, 17.0f);\n"
        "        SVPWMPack_Step(&svpwm_pack_st, u_svpwm_pack, dt, y_svpwm_pack);\n"
        "    }"
    )


class DB42S02SVPWMBlock(SVPWMBlock):
    """SVPWMBlock with DB42S02 / AURIX CodeGen metadata."""

    C_SOURCES    = ["embed_sim_sv_pwm.c"]
    C_HEADERS    = ["embed_sim_sv_pwm.h"]
    C_CUSTOM_EMIT = (
        "    /* --- svpwm (SVPWMBlock) --- */\n"
        "    {\n"
        "        SVM_DutyCycle_Type  svm_duty;\n"
        "        real32_T            y_svpwm[4];\n"
        "        SVM_CalculateDutyCycle(y_svpwm_pack[0],\n"
        "                               y_svpwm_pack[1],\n"
        "                               &svm_duty);\n"
        "        SVM_GetDutyCyclesFloat(&svm_duty,\n"
        "                               &y_svpwm[0], &y_svpwm[1], &y_svpwm[2]);\n"
        "        y_svpwm[3] = (real32_T)svm_duty.sector;\n"
        "    }"
    )


# =============================================================================

def build_and_run() -> dict:
    """Run closed-loop simulation and emit AURIX C code."""

    print("=" * 68)
    print("  NANOTEC DB42S02  —  MPC FOC + SMO  |  AURIX TC3xx")
    print("=" * 68)
    print(f"  Target : {TARGET_RPM:.0f} RPM  |  Vdc={V_DC}V  "
          f"dt={DT*1e6:.0f}µs  T_sim={T_SIM}s")
    print(f"  MPC    : N={MPC_N}  Q_id={MPC_Q_ID:.1f}  Q_iq={MPC_Q_IQ:.1f}  "
          f"Q_omega={MPC_Q_OMEGA:.1f}  R_vd={MPC_R_VD:.4f}  R_vq={MPC_R_VQ:.4f}")
    print(f"  SMO    : k={MPC_SMO_K:.2f} V  fc={MPC_SMO_FC:.0f} Hz")
    print(f"  Load   : 0 → {T_LOAD_LIGHT*1e3:.0f} mN·m @ {T_LOAD_T1}s"
          f" → {T_LOAD_HEAVY*1e3:.0f} mN·m @ {T_LOAD_T2}s")
    if ENABLE_NOISE:
        print(f"  Noise  : ia/ib/ic σ={NOISE_I_SIGMA*1e3:.1f} mA  "
              f"theta_m σ={NOISE_THETA_SIGMA*1e3:.1f} mrad  seed={NOISE_SEED}")
    else:
        print(f"  Noise  : DISABLED  (set ENABLE_NOISE=True to activate)")
    print("=" * 68)

    # ── CodeGen region (code that runs on AURIX) ──────────────────────────────
    cg_start = CodeGenStart("cg_start")

    mpc = MPCControllerBlock(
        "mpc",
        # Motor parameters (from _DB42S02)
        P_POLES     = MPC_MOTOR.P_POLES,
        R_S         = MPC_MOTOR.R_S,
        L           = MPC_MOTOR.L_D,      # Ld = Lq for surface PMSM
        LAMBDA_PM   = MPC_MOTOR.LAMBDA_PM,
        J           = MPC_MOTOR.J_ROTOR,
        B           = MPC_MOTOR.B_FRICTION,
        I_MAX       = MPC_MOTOR.I_MAX,
        V_MAX       = MPC_MOTOR.V_MAX,
        # MPC parameters
        N           = MPC_N,
        Q_id        = MPC_Q_ID,
        Q_iq        = MPC_Q_IQ,
        Q_omega     = MPC_Q_OMEGA,
        R_vd        = MPC_R_VD,
        R_vq        = MPC_R_VQ,
        dt_s        = DT,
        # SMO parameters
        SMO_K       = MPC_SMO_K,
        SMO_FC      = MPC_SMO_FC,
        use_c_backend = False,
    )

    svpwm_pack = DB42S02SVPWMPackBlock("svpwm_pack", v_dc=V_DC, use_c_backend=False)
    svpwm      = DB42S02SVPWMBlock("svpwm", use_c_backend=False)
    cg_end     = CodeGenEnd("cg_end")

    # ── Simulation-only blocks ────────────────────────────────────────────────
    speed_ref    = VectorStep("speed_ref", step_time=0.0,
                               before_value=TARGET_RADS_MECH,
                               after_value=TARGET_RADS_MECH)
    motor        = DB42S02PlantBlock("motor")
    motor_delay  = VectorDelay("motor_delay", initial=[0.0] * _MOTOR_OUT_SIZE)
    ctrl         = CtrlPacker("ctrl_packer")
    sink         = VectorEnd("sink")
    sink_cg      = VectorEnd("sink_cg")

    # ── Wiring ────────────────────────────────────────────────────────────────
    # CodeGen chain: ctrl → cg_start → mpc → svpwm_pack → svpwm → cg_end
    cg_start >> mpc >> svpwm_pack >> svpwm >> cg_end

    # Plant receives duties directly from SVPWMBlock — not from cg_end.
    # Pattern from db42s02_openloop_fmu.py: svpwm >> motor directly.
    # cg_end is the CodeGen boundary only; routing through it adds a
    # step delay and output repacking that corrupts the duty signals.
    svpwm       >> motor

    # Feedback path: plant output → 1-step delay → CtrlPacker → cg_start
    motor       >> motor_delay
    motor_delay >> ctrl
    speed_ref   >> ctrl
    ctrl        >> cg_start

    motor       >> sink
    cg_end      >> sink_cg

    # ── Simulate ──────────────────────────────────────────────────────────────
    sim = EmbedSim(sinks=[sink, sink_cg], T=T_SIM, dt=DT,
                   solver=ODESolver.RK4)

    print("\n[Topology]")
    sim.topo.print_console()
    sim.topo.export_html(
        str(_HERE / "db42s02_mpc_topology.html"),
        wire_labels={
            ("speed_ref",   "ctrl_packer"): "ω_ref [rad/s]",
            ("motor",       "motor_delay"): "FMU feedback",
            ("motor_delay", "ctrl_packer"): "z⁻¹ feedback",
            ("ctrl_packer", "cg_start"):    "[ω_ref,θ_m,ia,ib,ic,ω_m]",
            ("cg_start",    "mpc"):         "sensor inputs",
            ("mpc",         "svpwm_pack"):  "[v_α,v_β]",
            ("svpwm_pack",  "svpwm"):       "[Vref,α]",
            ("svpwm",       "cg_end"):      "[ta,tb,tc,sector]",
            ("svpwm",       "motor"):       "[ta,tb,tc,sector]",
        })

    sim.scope.add(mpc,        indices=[0, 1],             label="Vab")
    sim.scope.add(svpwm_pack, indices=[0],                label="Vref")
    sim.scope.add(svpwm,      indices=[0, 1, 2, 3],       label="Duties")
    # FMU output bus: [0]=rpm [1]=ia [2]=ib [3]=ic [4]=theta_m
    #                 [5]=T_em [6]=id_out [7]=iq_out
    sim.scope.add(motor,      indices=[0, 5, 6, 7],       label="Motor")

    print("\nRunning simulation…")
    sim.run()
    print(f"  Done: {len(sim.scope.t)} steps")

    # ── CodeGen ───────────────────────────────────────────────────────────────
    print("\n[CodeGen] Generating AURIX C code…")
    result = cg_end.generate_step(
        cg_start    = cg_start,
        output_dir  = _ROOT,
        dt_hz       = 1.0 / DT,
        prefix      = "EmbedSim",
        write_files = True,
    )
    if result:
        gen = _ROOT / "embedsim_gen"
        print(f"  {gen}/embedsim_step.h")
        print(f"  {gen}/embedsim_step.c")
        print(f"  Input_T  : omega_ref_mech, theta_m, ia, ib, ic")
        print(f"  Output_T : ta, tb, tc, sector")

    # ── Extract scope data ────────────────────────────────────────────────────
    sc = sim.scope
    t  = np.array(sc.t, dtype=np.float32)
    ld = mpc.log_data

    def _s(label, pos):
        sig = sc.get_signal(label, pos)
        return sig if sig is not None else np.zeros(len(t), dtype=np.float32)

    def _i(key):
        if len(ld["t"]) > 1:
            return np.interp(t, ld["t"], ld[key]).astype(np.float32)
        return np.zeros(len(t), dtype=np.float32)

    def _m(pos):
        sig = sc.get_signal("Motor", pos)
        return sig if sig is not None else np.zeros(len(t), dtype=np.float32)

    return {
        "t":             t,
        "speed_rpm":     _m(0),   # FMU rpm       (scope Motor pos 0)
        "omega_ref_rpm": _i("speed_ref"),
        "iq_ref":        _i("iq_ref"),
        "iq":            _i("iq"),
        "id":            _i("id"),
        "v_alpha":       _s("Vab",    0),
        "v_beta":        _s("Vab",    1),
        "vref":          _s("Vref",   0),
        "ta":            _s("Duties", 0),
        "tb":            _s("Duties", 1),
        "tc":            _s("Duties", 2),
        "sector":        _s("Duties", 3),
        "torque":        _m(1),   # FMU T_em      (scope Motor pos 1)
        "id_plant":      _m(2),   # FMU id_out    (scope Motor pos 2)
        "iq_plant":      _m(3),   # FMU iq_out    (scope Motor pos 3)
    }


# =============================================================================
# Plot
# =============================================================================

def plot_results(d: dict,
                 path: str = "db42s02_mpc_foc_20k_results.png") -> None:
    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    fig.suptitle(
        f"NANOTEC DB42S02 — MPC FOC + SMO  |  {TARGET_RPM:.0f} RPM  |  20 kHz"
        + ("  |  NOISE ON" if ENABLE_NOISE else "  |  no noise"),
        fontsize=12, fontweight="bold")
    t = d["t"]

    ax = axes[0, 0]
    ax.plot(t, d["omega_ref_rpm"], "k--", lw=1.5, label="ω_ref")
    ax.plot(t, d["speed_rpm"],     "C0",  lw=1.5, label="ω_actual")
    ax.axvline(T_LOAD_T1, color="orange", ls=":", lw=1.0)
    ax.axvline(T_LOAD_T2, color="red",    ls=":", lw=1.0)
    ax.set_ylabel("Speed [RPM]"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3); ax.set_title("Speed tracking"); ax.set_xlabel("t [s]")

    ax = axes[0, 1]
    ax.plot(t, d["speed_rpm"] - d["omega_ref_rpm"], "C1", lw=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(T_LOAD_T1, color="orange", ls=":", lw=1.0)
    ax.axvline(T_LOAD_T2, color="red",    ls=":", lw=1.0)
    ax.set_ylabel("Error [RPM]"); ax.grid(alpha=0.3)
    ax.set_title("Speed error"); ax.set_xlabel("t [s]")

    ax = axes[1, 0]
    ax.plot(t, d["iq_ref"], "k--", lw=1.2, label="iq_ref")
    ax.plot(t, d["iq"],     "C0",  lw=1.0, label="iq_meas")
    ax.plot(t, d["id"],     "C5",  lw=1.0, label="id_meas")
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.axhline( MPC_MOTOR.I_MAX, color="gray", ls="--", lw=0.5, alpha=0.5)
    ax.axhline(-MPC_MOTOR.I_MAX, color="gray", ls="--", lw=0.5, alpha=0.5)
    ax.set_ylabel("Current [A]"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3); ax.set_title("dq currents (MTPA  id_ref=0)")
    ax.set_xlabel("t [s]")

    ax = axes[1, 1]
    ax.plot(t, d["id"], "C5", lw=0.8)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_ylabel("id [A]"); ax.grid(alpha=0.3)
    ax.set_title("id  (should ≈ 0  —  MTPA)"); ax.set_xlabel("t [s]")

    ax = axes[2, 0]
    ax.plot(t, d["v_alpha"], "C0", lw=0.8, label="v_α")
    ax.plot(t, d["v_beta"],  "C1", lw=0.8, label="v_β")
    ax.set_ylabel("Voltage [V]"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3); ax.set_title("Stator voltage commands")
    ax.set_xlabel("t [s]")

    ax = axes[2, 1]
    ax.plot(t, d["vref"], "C5", lw=0.8)
    ax.axhline(0.95, color="red", ls="--", lw=0.8, alpha=0.7, label="clip=0.95")
    ax.set_ylabel("Vref [norm]"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3); ax.set_title("SVPWM modulation index")
    ax.set_xlabel("t [s]")

    ax = axes[3, 0]
    ax.plot(t, d["ta"], "C3", lw=0.7, label="ta")
    ax.plot(t, d["tb"], "C2", lw=0.7, label="tb")
    ax.plot(t, d["tc"], "C1", lw=0.7, label="tc")
    ax.set_ylim(-0.05, 1.05); ax.set_ylabel("Duty")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("SVPWM duties"); ax.set_xlabel("t [s]")

    ax = axes[3, 1]
    ax.plot(t, d["torque"] * 1000, "C4", lw=0.8, label="T_em")
    ax.axhline(T_LOAD_LIGHT * 1000, color="orange", ls=":", lw=1.0,
               alpha=0.7, label=f"{T_LOAD_LIGHT*1e3:.0f} mN·m")
    ax.axhline(T_LOAD_HEAVY * 1000, color="red",    ls=":", lw=1.0,
               alpha=0.7, label=f"{T_LOAD_HEAVY*1e3:.0f} mN·m")
    ax.set_ylabel("Torque [mN·m]"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3); ax.set_title("Electromagnetic torque")
    ax.set_xlabel("t [s]")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {path}")


# =============================================================================
# Summary
# =============================================================================

def print_summary(d: dict) -> None:
    t   = d["t"]
    rpm = d["speed_rpm"]
    ref = d["omega_ref_rpm"]
    iq  = d["iq"]
    n   = len(t)
    ss  = int(0.80 * n)

    ss_err  = float(np.mean(np.abs(rpm[ss:] - ref[ss:])))
    vref_mx = float(np.max(d["vref"]))
    iq_ss   = float(np.mean(np.abs(iq[ss:])))

    after   = t > T_LOAD_T2
    if np.any(after):
        pre     = t < T_LOAD_T2
        spd_pre = float(np.mean(rpm[pre][-50:])) if np.any(pre) else 0.0
        drop    = max(0.0, spd_pre - float(np.mean(rpm[after][:50])))
    else:
        drop = 0.0

    print("\n" + "=" * 60)
    print("  MPC FOC + SMO — Performance Summary")
    print("=" * 60)
    print(f"  Final speed    : {rpm[-1]:.1f} RPM  (target {TARGET_RPM:.0f})")
    print(f"  SS error       : {ss_err:.2f} RPM  (last 20%)")
    print(f"  Load drop      : {drop:.0f} RPM  at t={T_LOAD_T2}s")
    print(f"  iq_ref SS      : {iq_ss:.3f} A  "
          f"(expected {T_LOAD_HEAVY/MPC_MOTOR.KT:.2f} A for 20 mN·m)")
    print(f"  Vref max       : {vref_mx:.3f}  (clip 0.95)")
    print("=" * 60)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    data = build_and_run()
    plot_results(data)
    print_summary(data)
    print("\n[Done]")
    print("  db42s02_mpc_foc_20k_results.png")
    print("  db42s02_mpc_topology.html")
    print("  embedsim_gen/embedsim_step.c   ← flash to AURIX")
    print("  embedsim_gen/embedsim_step.h")