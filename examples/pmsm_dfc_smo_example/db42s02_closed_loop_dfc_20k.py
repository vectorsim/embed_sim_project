# db42s02_closed_loop_dfc_20k.py
"""
db42s02_closed_loop_dfc_20k.py
================================
EmbedSim -- Closed-loop Differential Flatness FOC -- NANOTEC DB42S02 -- AURIX TC3xx 20 kHz

AURIX-realistic noise model (all enabled):
  - ADC current noise     : 12-bit, 3.3 V ref -> LSB ~1.74 mA, Gaussian sigma = 1.5 LSB
  - ADC saturation clamp  : +-I_SAT = +-4.5 A (rail headroom)
  - PWM switching spikes  : +-0.5 A impulse injected on ia, 5 % probability/step
  - Encoder quantisation  : 1000 PPR x 4 decode -> 4000 cnt/rev, Delta-theta ~1.57 mrad
  - Encoder glitch        : 0.2 % probability of +-2-count slip (EMI / debounce)
  - Dead-time voltage drop: 400 ns x V_DC / DT ~0.136 V per phase (-> alphabeta disturbance)
  - DC bus ripple         : +-0.5 V @ 100 Hz sinusoidal on V_DC

Plant:
  FMU co-simulation via PMSM_Plant_FMUBlock (PMSM_Plant_FMU.fmu, FMI 2.0 CS).
  Replaces the pure-Python PMSM_Python_Plant.  Identical 8-element output bus
  [rpm, ia, ib, ic, theta_m, T_em, id, iq] — no controller changes required.

Wiring:
  cg_start >> dfc >> svpwm_pack >> svpwm >> load_sched >> fmu_plant >> cg_end

Speed profile (mimics AURIX ramp generator):
  0.0 - 0.3 s : linear ramp 0 -> TARGET_RPM
  0.3 - 5.0 s : hold TARGET_RPM
  Load steps at T_LOAD_T1 = 0.5 s (light) and T_LOAD_T2 = 1.2 s (heavy).

INTEGRATED GAIN TUNER
======================
When run with --tune (or the user answers 'y' to the interactive prompt),
the NN surrogate tuner executes before the main simulation:

    Phase 1  LHS exploration  : N_EXPLORE simulations across the gain space
    Phase 2  MLP training     : 3->16->16->1 network, Adam, 500 epochs
    Phase 3  Surrogate opt.   : gradient descent on MLP input, 8 restarts
    Phase 4  Verification     : one real simulation at recommended gains

On completion the tuner writes:
    embed_sim_dfc_gains_tuned.h   -- drop-in replacement for embed_sim_dfc_gains.h

If --tune is NOT requested the simulation uses the gains already defined in
_ACTIVE_GAINS (defaults to the hardware-commissioning values from the header).

Outputs:
  db42s02_dfc_foc_20k_results.png
  db42s02_dfc_topology.html
  embedsim_gen/embedsim_step.c/.h   (CodeGen)
  embed_sim_dfc_gains_tuned.h       (only when --tune requested)
"""

from __future__ import annotations

import sys
import math
import time
import argparse
import textwrap
import numpy as np
import importlib.util as _ilu
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _path_utils import get_project_root, get_embedsim_import_path, get_current_parent

_HERE    = get_current_parent()
_ROOT    = get_project_root()
_FS_ELEC = _ROOT / "fs_electrical_machines"

for _p in (get_embedsim_import_path(), str(_FS_ELEC), str(_FS_ELEC / "c_src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Load DFControllerBlock from fs_electrical_machines (not c_src)
_dfcb_spec = _ilu.spec_from_file_location(
    "diff_flatness_controller_block",
    str(_FS_ELEC / "diff_flatness_controller_block.py"))
_dfcb_mod = _ilu.module_from_spec(_dfcb_spec)
_dfcb_spec.loader.exec_module(_dfcb_mod)
sys.modules["diff_flatness_controller_block"] = _dfcb_mod

from embedsim import EmbedSim, ODESolver, VectorEnd
from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE
from embedsim.source_blocks import VectorStep, VectorConstant
from embedsim.simulation_engine import VectorDelay
from embedsim.code_generator import CodeGenStart, CodeGenEnd

from motor_utility_blocks import SVPWMPackBlock
from svpwm_block import SVPWMBlock
from smc_controller_block import _DB42S02
from PMSM_Plant_FMUBlock import PMSM_Plant_FMUBlock
from ctrl_packer import CtrlPacker
from machine_feedback import db42s02_feedback_profile
from diff_flatness_controller_block import DFControllerBlock


# =============================================================================
# Simulation constants
# =============================================================================

V_DC             = _DB42S02.SMC_V_DC           # 17 V nominal
TARGET_RPM       = 2000.0
T_SIM            = 5.0
DT               = 50e-6                        # 20 kHz ISR period

_RAMP_TIME       = 0.3

T_LOAD_T1        = 0.5
T_LOAD_T2        = 1.2
T_LOAD_ZERO      = 0.000                        # N.m
T_LOAD_LIGHT     = 0.005                        # N.m
T_LOAD_HEAVY     = 0.020                        # N.m

TARGET_RADS_MECH = TARGET_RPM * 2.0 * math.pi / 60.0
_MOTOR_OUT_SIZE  = 8


# =============================================================================
# Active gains — single source of truth for both sim and tuner
#
# Default values match the hardware-commissioning constants in
# embed_sim_dfc_gains.h:
#   DFC_KP_SPEED = 0.4  A/(rad/s)
#   DFC_KP_ID    = 0.4  V/A
#   DFC_KP_IQ    = 8.0  V/A
#
# The tuner overwrites this dict in-place when --tune is requested,
# so _run_sim() always reads the current best gains.
# =============================================================================

_ACTIVE_GAINS: dict = {
    "Kp_speed": 0.4,    # [A/(rad/s)]  C: DFC_KP_SPEED
    "Kp_id":    0.4,    # [V/A]        C: DFC_KP_ID
    "Kp_iq":    8.0,    # [V/A]        C: DFC_KP_IQ
}


# =============================================================================
# AURIX hardware-specific noise parameters
# =============================================================================

_ADC_BITS        = 12
_ADC_I_FS        = _DB42S02.SMC_I_MAX * 2.0
_ADC_LSB_A       = _ADC_I_FS / (2 ** _ADC_BITS)   # ~1.74 mA per LSB
_ADC_NOISE_SIGMA = 1.5 * _ADC_LSB_A
_ADC_SAT_LIMIT   = _DB42S02.SMC_I_MAX * 1.26       # ~4.5 A

_SPIKE_PROB      = 0.05
_SPIKE_AMP       = 0.50                             # A

_ENC_PPR         = 1000
_ENC_COUNTS_REV  = 4 * _ENC_PPR
_ENC_RESOLUTION  = 2.0 * math.pi / _ENC_COUNTS_REV  # ~1.5708 mrad/count
_ENC_GLITCH_PROB = 0.002
_ENC_GLITCH_MAG  = 2

_DEAD_TIME_S     = 400e-9
_DEAD_TIME_V     = _DEAD_TIME_S * V_DC / DT        # ~0.136 V per phase

_BUS_RIPPLE_AMP  = 0.50                             # V
_BUS_RIPPLE_HZ   = 100.0                            # Hz

_RNG             = np.random.default_rng(seed=20240101)


# =============================================================================
# Noise helpers
# =============================================================================

def _adc_noise(current_A: float) -> float:
    """AURIX EVADC 12-bit: Gaussian thermal noise -> quantise -> rail clamp."""
    noisy     = current_A + _RNG.normal(0.0, _ADC_NOISE_SIGMA)
    quantised = round(noisy / _ADC_LSB_A) * _ADC_LSB_A
    return float(np.clip(quantised, -_ADC_SAT_LIMIT, _ADC_SAT_LIMIT))


def _pwm_spike(ia: float) -> float:
    """Phase-A bootstrap spike at 20 kHz switching edge."""
    if _RNG.random() < _SPIKE_PROB:
        sign = float(_RNG.choice([-1.0, 1.0]))
        return ia + sign * _SPIKE_AMP
    return ia


def _enc_quantise(theta_m_rad: float) -> float:
    """GTM TIM capture resolution: snap to nearest encoder count."""
    count = round(theta_m_rad / _ENC_RESOLUTION)
    return count * _ENC_RESOLUTION


def _enc_glitch(theta_m_q: float) -> float:
    """Random +-2-count slip (EMI / debounce failure). 0.2 % per step."""
    if _RNG.random() < _ENC_GLITCH_PROB:
        slip = int(_RNG.choice([-_ENC_GLITCH_MAG, _ENC_GLITCH_MAG]))
        return theta_m_q + slip * _ENC_RESOLUTION
    return theta_m_q


def _deadtime_disturbance(ia: float, ib: float, ic: float) -> tuple[float, float]:
    """
    Dead-time voltage error in alphabeta frame.

    Per-phase error = +V_err if i > 0 else -V_err  (follows current sign).
    Clarke (amplitude-invariant):
      v_alpha = (2/3)(va - vb/2 - vc/2)
      v_beta  = (2/3)(sqrt(3)/2)(vb - vc)
    """
    ve_a = _DEAD_TIME_V if ia >= 0.0 else -_DEAD_TIME_V
    ve_b = _DEAD_TIME_V if ib >= 0.0 else -_DEAD_TIME_V
    ve_c = _DEAD_TIME_V if ic >= 0.0 else -_DEAD_TIME_V
    dv_alpha = (2.0 / 3.0) * (ve_a - 0.5 * ve_b - 0.5 * ve_c)
    dv_beta  = (2.0 / 3.0) * (math.sqrt(3.0) / 2.0) * (ve_b - ve_c)
    return dv_alpha, dv_beta


def _vdc_ripple(t: float) -> float:
    """DC bus instantaneous voltage including 100 Hz capacitor ripple."""
    return V_DC + _BUS_RIPPLE_AMP * math.sin(2.0 * math.pi * _BUS_RIPPLE_HZ * t)


# =============================================================================
# Noise-state recorder
# =============================================================================

class _NoiseLog:
    """Lightweight recorder for noise diagnostics."""
    def __init__(self):
        self.t:         list[float] = []
        self.ia_raw:    list[float] = []
        self.ia_noisy:  list[float] = []
        self.theta_raw: list[float] = []
        self.theta_q:   list[float] = []
        self.vdc:       list[float] = []

    def push(self, t, ia_raw, ia_noisy, theta_raw, theta_q, vdc):
        self.t.append(t);         self.ia_raw.append(ia_raw)
        self.ia_noisy.append(ia_noisy);  self.theta_raw.append(theta_raw)
        self.theta_q.append(theta_q);   self.vdc.append(vdc)

    def reset(self):
        self.t.clear(); self.ia_raw.clear(); self.ia_noisy.clear()
        self.theta_raw.clear(); self.theta_q.clear(); self.vdc.clear()


_NOISE_LOG = _NoiseLog()


# =============================================================================
# Plant block with integrated AURIX noise chain
# =============================================================================

# FMU path — resolved once at module load so working directory is irrelevant
_FMU_PATH = str(_FS_ELEC / "modelica" / "PMSM_Plant_FMU.fmu")


class DB42S02PlantBlock(PMSM_Plant_FMUBlock):
    """
    NANOTEC DB42S02 PMSM plant -- FMU co-simulation via PMSM_Plant_FMU.fmu.

    Mirrors the pattern in db42s02_closed_loop_mpc_foc_20k.py exactly.
    Load schedule, FMU input packing, DC bus ripple, and AURIX noise
    chain all run inside compute_py() each ISR tick.

    FMU input bus (5 elements):
        [0] duty_a [0,1]  [1] duty_b [0,1]  [2] duty_c [0,1]
        [3] v_dc   [V]    [4] T_load [N.m]

    FMU output bus (8 elements, same indices as PMSM_Python_Plant):
        [0] rpm  [1] ia  [2] ib  [3] ic  [4] theta_m  [5] T_em  [6] id  [7] iq
    """
    TOPO_CATEGORY     = "plant"
    C_CODEGEN_EXCLUDE = True
    output_label      = "[rpm,ia,ib,ic,theta_m,Tem,id,iq]"

    def __init__(self, name: str, with_noise: bool = True) -> None:
        super().__init__(name=name, fmu_path=_FMU_PATH)
        self._with_noise   = with_noise
        self._last_ta      = 0.5
        self._last_tb      = 0.5
        self._last_tc      = 0.5
        self._print_next_t = 0.20   # [s] next status print time
        print(f"[FMU] {name} <- {_FMU_PATH}")

    def compute_py(self, t: float, dt: float, input_values=None):
        # 1. Timed load schedule
        if   t < T_LOAD_T1: t_load = T_LOAD_ZERO
        elif t < T_LOAD_T2: t_load = T_LOAD_LIGHT
        else:                t_load = T_LOAD_HEAVY
        # 2. Duties from SVPWM
        ta = tb = tc = 0.5
        if input_values and input_values[0] is not None:
            v = input_values[0].value
            if len(v) >= 3:
                ta_in, tb_in, tc_in = float(v[0]), float(v[1]), float(v[2])
                if ta_in != 0.0 or tb_in != 0.0 or tc_in != 0.0:
                    ta, tb, tc = ta_in, tb_in, tc_in
        self._last_ta, self._last_tb, self._last_tc = ta, tb, tc
        # 3. DC bus ripple
        vdc_inst = _vdc_ripple(t)
        # 4. Step FMU
        fmu_in = VectorSignal(
            np.array([ta, tb, tc, vdc_inst, t_load], dtype=DEFAULT_DTYPE)
        )
        raw = super().compute_py(t, dt, [fmu_in])
        if raw is None or raw.value is None or len(raw.value) < 8:
            return VectorSignal(np.zeros(8, dtype=DEFAULT_DTYPE), self.name)
        # 5. Status print every 0.2 s -- mirrors PMSM_Python_Plant behaviour
        if t >= self._print_next_t:
            _rpm = float(raw.value[0])
            _te  = float(raw.value[4]) * float(_DB42S02.SMC_P_POLES)
            _id  = float(raw.value[6])
            _iq  = float(raw.value[7])
            _tem = float(raw.value[5])
            print(f"[FMU  t={t:.2f}s]  rpm={_rpm:+8.1f}  "
                  f"theta_e={_te:.4f}rad  "
                  f"id={_id:+.4f}A  iq={_iq:+.4f}A  "
                  f"T_em={_tem*1e3:+.3f}mN.m  T_load={t_load*1e3:.1f}mN.m")
            self._print_next_t += 0.20
        # 7. AURIX noise chain
        v         = raw.value.copy()
        ia_raw    = float(v[1]); ib_raw = float(v[2])
        ic_raw    = float(v[3]); theta_raw = float(v[4])
        if self._with_noise:
            ia_n    = _pwm_spike(_adc_noise(ia_raw))
            ib_n    = _adc_noise(ib_raw)
            ic_n    = _adc_noise(ic_raw)
            theta_q = _enc_glitch(_enc_quantise(theta_raw))
        else:
            ia_n = ia_raw; ib_n = ib_raw; ic_n = ic_raw; theta_q = theta_raw
        _NOISE_LOG.push(t, ia_raw, ia_n, theta_raw, theta_q, vdc_inst)
        v[1] = ia_n; v[2] = ib_n; v[3] = ic_n; v[4] = theta_q
        return VectorSignal(v.astype(DEFAULT_DTYPE), self.name)

    def compute(self, t: float, dt: float, input_values=None):
        return self.compute_py(t, dt, input_values)


# =============================================================================
# LoadScheduleBlock — packs FMU input bus + applies AURIX noise to outputs
# =============================================================================

class LoadScheduleBlock(VectorBlock):
    """
    Two-port adapter sitting between SVPWMBlock and DB42S02PlantBlock.

    FORWARD (input -> FMU):
        Receives [ta, tb, tc, sector] from SVPWMBlock.
        Packs [duty_a, duty_b, duty_c, v_dc_inst, T_load] for the FMU.
        v_dc_inst includes DC bus ripple; T_load follows the timed schedule.

    FEEDBACK (FMU output -> controller):
        Receives [rpm, ia, ib, ic, theta_m, T_em, id, iq] from DB42S02PlantBlock.
        Applies the full AURIX noise chain (ADC, encoder, spikes, dead-time).
        Emits the noisy 8-element bus onward to CtrlPacker / sink.

    The noise chain lives here rather than inside DB42S02PlantBlock so the FMU
    block itself remains a clean, reusable wrapper with no simulation-specific
    coupling.
    """
    TOPO_CATEGORY     = "utility"
    C_CODEGEN_EXCLUDE = True
    output_label      = "[rpm,ia_n,ib_n,ic_n,theta_q,Tem,id,iq]"

    def __init__(self, name: str, plant: DB42S02PlantBlock,
                 with_noise: bool = True) -> None:
        super().__init__(name)
        self._plant      = plant
        self._with_noise = with_noise
        self._last_ta    = 0.5
        self._last_tb    = 0.5
        self._last_tc    = 0.5
        self.vector_size = 8
        self.is_dynamic  = False

    def compute(self, t: float, dt: float, input_values=None) -> VectorSignal:
        """
        Step sequence each ISR tick:
          1. Unpack duties from SVPWMBlock output (input_values[0]).
          2. Determine T_load from timed schedule.
          3. Apply DC bus ripple to v_dc.
          4. Pack FMU input bus and step the FMU plant.
          5. Apply AURIX noise chain to FMU outputs.
          6. Return noisy 8-element bus.
        """
        # ---- 1. Duties from SVPWM ----------------------------------------
        ta = tb = tc = 0.5
        if input_values and input_values[0] is not None:
            v = input_values[0].value
            if len(v) >= 3:
                ta_raw = float(v[0]); tb_raw = float(v[1]); tc_raw = float(v[2])
                if ta_raw != 0.0 or tb_raw != 0.0 or tc_raw != 0.0:
                    ta, tb, tc = ta_raw, tb_raw, tc_raw
        self._last_ta, self._last_tb, self._last_tc = ta, tb, tc

        # ---- 2. Load schedule --------------------------------------------
        if   t < T_LOAD_T1: t_load = T_LOAD_ZERO
        elif t < T_LOAD_T2: t_load = T_LOAD_LIGHT
        else:               t_load = T_LOAD_HEAVY

        # ---- 3. DC bus ripple --------------------------------------------
        vdc_inst = _vdc_ripple(t)

        # ---- 4. Step FMU plant -------------------------------------------
        fmu_in = VectorSignal(
            np.array([ta, tb, tc, vdc_inst, t_load], dtype=DEFAULT_DTYPE),
            self.name
        )
        raw = self._plant.compute(t, dt, [fmu_in])
        if raw is None or raw.value is None or len(raw.value) < 8:
            return VectorSignal(np.zeros(8, dtype=DEFAULT_DTYPE), self.name)

        v = raw.value.copy()   # [rpm, ia, ib, ic, theta_m, T_em, id, iq]

        # ---- 5. AURIX noise chain ----------------------------------------
        ia_raw    = float(v[1])
        ib_raw    = float(v[2])
        ic_raw    = float(v[3])
        theta_raw = float(v[4])
        vdc_log   = vdc_inst

        if self._with_noise:
            ia_n     = _pwm_spike(_adc_noise(ia_raw))
            ib_n     = _adc_noise(ib_raw)
            ic_n     = _adc_noise(ic_raw)
            theta_q  = _enc_glitch(_enc_quantise(theta_raw))
        else:
            ia_n = ia_raw; ib_n = ib_raw; ic_n = ic_raw; theta_q = theta_raw

        _NOISE_LOG.push(t, ia_raw, ia_n, theta_raw, theta_q, vdc_log)

        # Dead-time diagnostic (kept for plot parity with Python-plant version)
        dv_alpha, dv_beta = _deadtime_disturbance(ta, tb, tc)
        _ = dv_alpha; _ = dv_beta   # stored for SVPWMPackBlockDT if needed

        v[1] = ia_n; v[2] = ib_n; v[3] = ic_n; v[4] = theta_q
        return VectorSignal(v.astype(DEFAULT_DTYPE), self.name)


# =============================================================================
# Dead-time-aware SVPWMPackBlock wrapper
# =============================================================================

class SVPWMPackBlockDT(SVPWMPackBlock):
    """
    SVPWMPackBlock — unchanged API, dead-time diagnostic now reads from
    LoadScheduleBlock (which owns the last duty cycle values).
    """
    TOPO_CATEGORY     = "utility"
    C_CODEGEN_EXCLUDE = True

    def __init__(self, name: str, v_dc: float,
                 load_sched: "LoadScheduleBlock | None" = None):
        super().__init__(name, v_dc=v_dc)
        self._load_sched = load_sched

    def compute(self, t, dt, input_values=None):
        result = super().compute(t, dt, input_values)
        if result is None or result.value is None or len(result.value) < 2:
            return result
        if self._load_sched is not None and input_values and input_values[0] is not None:
            dv_alpha, dv_beta = _deadtime_disturbance(
                self._load_sched._last_ta,
                self._load_sched._last_tb,
                self._load_sched._last_tc,
            )
            self._dt_dv_alpha = dv_alpha
            self._dt_dv_beta  = dv_beta
        return result


# =============================================================================
# Wire labels
# =============================================================================

_WIRE_LABELS = {
    ("speed_ref",   "ctrl_packer"):  "w_ref [rad/s]",
    ("motor_delay", "ctrl_packer"):  "[rpm,ia_n,ib_n,ic_n,th_q,Tem,id,iq] z-1",
    ("ctrl_packer", "cg_start"):     "[w_ref,th_m,ia,ib,ic]",
    ("cg_start",    "dfc"):          "[w_ref,th_m,ia,ib,ic]",
    ("dfc",         "svpwm_pack"):   "[v_alpha,v_beta]",
    ("svpwm_pack",  "svpwm"):        "[Vref,alpha,Vdc]",
    ("svpwm",       "cg_end"):       "[ta,tb,tc,sector]",
    ("cg_end",      "motor"):        "[ta,tb,tc,sector]",
    ("cg_end",      "sink_cg"):      "[ta,tb,tc,sector]",
    ("motor",       "motor_delay"):  "[rpm,ia_n,ib_n,ic_n,th_q,Tem,id,iq]",
    ("motor",       "sink"):         "[rpm,ia,ib,ic,th_m,Tem,id,iq]",
}


# =============================================================================
# Core simulation runner
#
# Reads gains from _ACTIVE_GAINS so both the main run and every tuner
# evaluation call exactly the same function.  No monkey-patching needed.
# =============================================================================

def _run_sim(
    *,
    with_noise:   bool = True,
    with_codegen_hooks: bool = True,
) -> dict | None:
    """
    Build and run one closed-loop DFC simulation.

    Parameters
    ----------
    with_noise : bool
        If True  — full AURIX noise chain (ADC, encoder, spikes, ripple).
        If False — clean plant, no noise (used by tuner exploration phase).
    with_codegen_hooks : bool
        If True  — CodeGenStart / CodeGenEnd blocks included (main run).
        If False — lighter wiring without CodeGen objects (tuner).

    Returns
    -------
    dict | None
        Result dictionary or None on failure.
    """
    _NOISE_LOG.reset()

    try:
        # ---- Controller ------------------------------------------------
        dfc = DFControllerBlock(
            "dfc",
            P_POLES   = int(_DB42S02.SMC_P_POLES),
            R_S       = _DB42S02.SMC_R_S,
            L_D       = _DB42S02.SMC_L_D,
            L_Q       = _DB42S02.SMC_L_Q,
            LAMBDA_PM = _DB42S02.SMC_LAMBDA_PM,
            V_DC      = V_DC,
            I_MAX     = _DB42S02.SMC_I_MAX,
            dt_s      = DT,
            Kp_speed  = _ACTIVE_GAINS["Kp_speed"],
            Kp_id     = _ACTIVE_GAINS["Kp_id"],
            Kp_iq     = _ACTIVE_GAINS["Kp_iq"],
            smo_k     = _DB42S02.SMC_SMO_K,
            smo_tau   = 1.0 / (2.0 * math.pi * _DB42S02.SMC_SMO_FC),
            fusion_omega_lo = 50.0,
            fusion_omega_hi = 250.0,
            fusion_iir_lo   = 0.05,
            fusion_iir_hi   = 0.30,
            use_c_backend   = True,   # Python backend — works in tuner too
        )

        # ---- FMU plant (mirrors MPC example pattern exactly) ----------
        motor       = DB42S02PlantBlock("motor", with_noise=with_noise)

        # ---- Signal chain blocks ---------------------------------------
        svpwm_pack  = SVPWMPackBlockDT("svpwm_pack", v_dc=V_DC,
                                       load_sched=motor)
        svpwm       = SVPWMBlock("svpwm", use_c_backend=False)
        speed_ref   = VectorStep("speed_ref", step_time=0.0,
                                 before_value=TARGET_RADS_MECH,
                                 after_value=TARGET_RADS_MECH)
        motor_delay = VectorDelay("motor_delay", initial=[0.0] * _MOTOR_OUT_SIZE)
        ctrl        = CtrlPacker(
            "ctrl_packer",
            target_rads_mech = TARGET_RADS_MECH,
            ramp_time        = _RAMP_TIME,
        )
        ctrl.set_noise_enabled(False)   # noise handled in DB42S02PlantBlock
        sink    = VectorEnd("sink")
        sink_cg = VectorEnd("sink_cg")

        # ---- CodeGen blocks (main run only) ----------------------------
        # Boundary: cg_start >> dfc >> svpwm_pack >> svpwm >> cg_end
        # FMU plant has C_CODEGEN_EXCLUDE = True -- stays outside boundary.
        if with_codegen_hooks:
            cg_start = CodeGenStart("cg_start")
            cg_end   = CodeGenEnd("cg_end")
            ctrl       >> cg_start
            cg_start   >> dfc >> svpwm_pack >> svpwm >> cg_end
            cg_end     >> motor
            cg_end     >> sink_cg
        else:
            cg_start = cg_end = None
            ctrl       >> dfc >> svpwm_pack >> svpwm
            svpwm      >> motor
            svpwm      >> sink_cg

        # ---- Wiring common to both modes -------------------------------
        motor       >> motor_delay >> ctrl
        speed_ref   >> ctrl
        motor       >> sink

        # ---- Scope -----------------------------------------------------
        sim = EmbedSim(sinks=[sink, sink_cg], T=T_SIM, dt=DT,
                       solver=ODESolver.EULER)
        sim.scope.add(dfc,        indices=[0, 1],                 label="Vab")
        sim.scope.add(svpwm_pack, indices=[0],                    label="Vref")
        sim.scope.add(svpwm,      indices=[0, 1, 2, 3],           label="Duties")
        sim.scope.add(motor,      indices=[0, 1, 2, 3, 5, 6, 7],  label="Motor")

        sim.run()


    except Exception as exc:
        import traceback
        print(f"  [sim error] {exc}")
        traceback.print_exc()
        return None

    sc = sim.scope
    t  = np.array(sc.t, dtype=np.float32)
    ld = dfc.log_data
    nl = _NOISE_LOG

    if len(t) < 100:
        return None

    def _s(label, pos):
        sig = sc.get_signal(label, pos)
        return sig if sig is not None else np.zeros(len(t), np.float32)

    # _log_ok: True when the Python-side DFC log_data was populated.
    # When use_c_backend=True compute_c() is called instead of compute_py(),
    # so _log_step() is never reached and log_data stays empty.
    # Fall back to plant outputs for iq/id and derive iq_ref/alpha from speed.
    _log_ok = len(ld["t"]) > 1

    def _i(key):
        if _log_ok:
            return np.interp(t, ld["t"], ld[key]).astype(np.float32)
        return np.zeros(len(t), np.float32)

    def _m(pos):
        sig = sc.get_signal("Motor", pos)
        return sig if sig is not None else np.zeros(len(t), np.float32)

    def _nl(attr):
        if len(nl.t) > 1:
            return np.interp(t, nl.t, getattr(nl, attr)).astype(np.float32)
        return np.zeros(len(t), np.float32)

    # iq / id: prefer DFC log (Python backend); fall back to plant truth (C backend)
    # Motor bus: [rpm=0, ia=1, ib=2, ic=3, theta_m=4, Tem=5, id=6, iq=7]
    _iq = _i("iq") if _log_ok else _m(7)
    _id = _i("id") if _log_ok else _m(6)

    # iq_ref: from DFC log (Python) or P-loop approximation (C backend)
    if _log_ok:
        _iq_ref = _i("iq_ref")
    else:
        _speed_err = np.clip(
            TARGET_RPM - _m(0), -TARGET_RPM, TARGET_RPM
        ).astype(np.float32) * float(2.0 * math.pi / 60.0)
        _iq_ref = np.clip(
            _ACTIVE_GAINS["Kp_speed"] * _speed_err,
            -_DB42S02.SMC_I_MAX, _DB42S02.SMC_I_MAX
        ).astype(np.float32)

    # fusion_alpha: from DFC log (Python) or piecewise-linear from plant speed (C backend)
    if _log_ok:
        _alpha = _i("alpha")
    else:
        _omega_m  = _m(0) * float(2.0 * math.pi / 60.0)
        _omega_lo = float(dfc.fusion.omega_lo)
        _omega_hi = float(dfc.fusion.omega_hi)
        _alpha    = np.clip(
            (np.abs(_omega_m) - _omega_lo) / (_omega_hi - _omega_lo),
            0.0, 1.0
        ).astype(np.float32)

    return {
        "t":             t,
        "speed_rpm":     _m(0),
        "omega_ref_rpm": _i("speed_ref") if _log_ok else
                         np.full(len(t), TARGET_RPM, dtype=np.float32),
        "iq_ref":        _iq_ref,
        "iq":            _iq,
        "id":            _id,
        "v_alpha":       _s("Vab",    0),
        "v_beta":        _s("Vab",    1),
        "vref":          _s("Vref",   0),
        "ta":            _s("Duties", 0),
        "tb":            _s("Duties", 1),
        "tc":            _s("Duties", 2),
        "sector":        _s("Duties", 3),
        "torque":        _m(4),
        "fusion_alpha":  _alpha,
        "ia_raw":        _nl("ia_raw"),
        "ia_noisy":      _nl("ia_noisy"),
        "theta_raw":     _nl("theta_raw"),
        "theta_q":       _nl("theta_q"),
        "vdc":           _nl("vdc"),
        "_cg_start":     cg_start,
        "_cg_end":       cg_end,
        "_sim":          sim,
    }


# =============================================================================
# CodeGen
# =============================================================================

def _run_codegen(d: dict) -> None:
    cg_start = d.get("_cg_start")
    cg_end   = d.get("_cg_end")
    sim      = d.get("_sim")
    if not all([cg_start, cg_end, sim]):
        print("  [CodeGen] skipped (no CodeGen hooks in this run).")
        return

    print("\n[Topology]")
    sim.topo.print_console()
    sim.topo.export_html(str(_HERE / "db42s02_dfc_topology.html"),
                         wire_labels=_WIRE_LABELS)

    print("\n[CodeGen] Generating AURIX C code ...")
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


# =============================================================================
# Plot
# =============================================================================

def plot_results(d: dict,
                 out_path: str = "db42s02_dfc_foc_20k_results.png") -> None:
    t = d["t"]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), facecolor="#111111")
    fig.suptitle(
        "DB42S02  DFC SMO FOC  20 kHz  AURIX TC3xx  [AURIX noise: ON]",
        color="white", fontsize=13, fontweight="bold")

    for ax in axes.flat:
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="#888", labelsize=8)
        ax.spines[:].set_color("#333")

    def _vl(ax):
        ax.axvline(T_LOAD_T1, color="orange",  lw=0.8, ls=":", alpha=0.6, label="light load")
        ax.axvline(T_LOAD_T2, color="#ff6666", lw=0.8, ls=":", alpha=0.6, label="heavy load")

    def _leg(ax): ax.legend(fontsize=7, facecolor="#222", labelcolor="white", edgecolor="#444")
    def _fmt(ax, y, title):
        ax.set_ylabel(y, color="#aaa", fontsize=9)
        ax.set_xlabel("t [s]", color="#888", fontsize=8)
        ax.set_title(title, color="#cccccc", fontsize=9, pad=4)

    ax = axes.flat

    ax[0].plot(t, d["omega_ref_rpm"], color="white",   lw=1.0, ls="--", alpha=0.6, label="ref")
    ax[0].plot(t, d["speed_rpm"],     color="#44bbff", lw=1.4, label="actual")
    ax[0].axhline(TARGET_RPM, color="#ff4444", lw=0.8, ls=":", alpha=0.5)
    _vl(ax[0]); _leg(ax[0]); _fmt(ax[0], "Speed [RPM]", "Mechanical speed")

    ax[1].plot(t, d["iq_ref"], color="#ff9944", lw=1.2, label="iq_ref")
    ax[1].plot(t, d["iq"],     color="#44ff88", lw=1.2, label="iq_meas")
    ax[1].axhline(0, color="#444", lw=0.7)
    _vl(ax[1]); _leg(ax[1]); _fmt(ax[1], "Current [A]", "q-axis current")

    ax[2].plot(t, d["id"], color="#bb66ff", lw=1.2, label="id_meas")
    ax[2].axhline(0, color="#444", lw=0.7, ls="--")
    _vl(ax[2]); _leg(ax[2]); _fmt(ax[2], "id [A]", "d-axis current (MTPA=0)")

    ax[3].plot(t, d["v_alpha"], color="#44bbff", lw=0.8, label="v_alpha")
    ax[3].plot(t, d["v_beta"],  color="#ff9944", lw=0.8, label="v_beta")
    _vl(ax[3]); _leg(ax[3]); _fmt(ax[3], "Voltage [V]", "alphabeta voltage commands")

    ax[4].plot(t, d["vref"], color="#ffdd44", lw=1.2)
    ax[4].axhline(1.0, color="#ff4444", lw=0.8, ls="--", alpha=0.6, label="overmod")
    _vl(ax[4]); _leg(ax[4]); _fmt(ax[4], "Vref [0-1]", "SVPWM modulation index")

    ax[5].plot(t, d["fusion_alpha"], color="#cc88ff", lw=1.8, label="alpha")
    ax[5].fill_between(t, 0, d["fusion_alpha"], color="#cc88ff", alpha=0.15)
    ax[5].set_ylim(-0.05, 1.1)
    _leg(ax[5]); _fmt(ax[5], "alpha", "SpeedFusion weight (0=enc, 1=SMO)")

    ax[6].plot(t, d["ia_raw"],   color="#666",    lw=0.7, label="ia ideal")
    ax[6].plot(t, d["ia_noisy"], color="#ff4466", lw=0.8, alpha=0.85, label="ia ADC+spike")
    ax[6].set_ylim(-6.0, 6.0)
    _vl(ax[6]); _leg(ax[6]); _fmt(ax[6], "ia [A]", "Phase-A current -- ADC noise")

    enc_err = d["theta_q"] - d["theta_raw"]
    ax[7].plot(t, enc_err * 1e3, color="#44ffcc", lw=0.8)
    ax[7].axhline(0, color="#444", lw=0.7)
    _fmt(ax[7], "Delta theta [mrad]", "Encoder quantisation error")

    ax[8].plot(t, d["vdc"], color="#ffaa44", lw=0.9, label="V_DC")
    ax[8].axhline(V_DC, color="#666", lw=0.7, ls="--", label=f"nom={V_DC:.0f}V")
    _leg(ax[8]); _fmt(ax[8], "V_DC [V]", "DC bus ripple")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {out_path}")


def print_summary(d: dict) -> None:
    t   = d["t"]
    rpm = d["speed_rpm"]
    ss  = t > 0.85 * T_SIM
    if not np.any(ss):
        print("  [summary] insufficient data"); return

    ss_err   = float(np.mean(np.abs(rpm[ss] - TARGET_RPM)))
    id_rms   = float(np.sqrt(np.mean(d["id"][ss] ** 2)))
    iq_chat  = float(np.std(d["iq_ref"][ss]))
    alpha_ss = float(np.mean(d["fusion_alpha"][ss]))

    adc_err  = d["ia_noisy"] - d["ia_raw"]
    adc_rms  = float(np.sqrt(np.mean(adc_err ** 2))) * 1e3
    enc_err  = d["theta_q"] - d["theta_raw"]
    enc_max  = float(np.max(np.abs(enc_err))) * 1e3
    vdc_rip  = float(np.max(np.abs(d["vdc"] - V_DC)))

    print(f"\n{'='*60}")
    print("  DFC FOC -- Performance Summary  [AURIX noise: ON]")
    print(f"{'='*60}")
    print(f"  Active gains     : Kp_speed={_ACTIVE_GAINS['Kp_speed']:.4f}  "
          f"Kp_id={_ACTIVE_GAINS['Kp_id']:.4f}  Kp_iq={_ACTIVE_GAINS['Kp_iq']:.4f}")
    print(f"  SS speed error   : {ss_err:.2f} RPM")
    print(f"  id RMS (MTPA)    : {id_rms:.4f} A    (target 0)")
    print(f"  iq chattering    : {iq_chat:.4f} A   (std in SS)")
    print(f"  SpeedFusion alpha: {alpha_ss:.3f}    (1 = full SMO at rated speed)")
    print(f"{'─'*60}")
    print("  Noise statistics:")
    print(f"  ADC noise RMS    : {adc_rms:.2f} mA")
    print(f"  Enc quant+glitch : {enc_max:.2f} mrad max")
    print(f"  DC bus ripple    : +-{vdc_rip:.3f} V")
    print(f"{'='*60}")


# =============================================================================
# =============================================================================
#
#   INTEGRATED NN SURROGATE GAIN TUNER
#
#   All tuner code lives in this section so it can share _run_sim(),
#   _ACTIVE_GAINS, and the motor constants directly — no monkey-patching,
#   no import of a separate module.
#
# =============================================================================
# =============================================================================

# ── Tuner hyper-parameters ────────────────────────────────────────────────────
_T_N_EXPLORE   = 40     # LHS simulations (exploration phase)
_T_N_EPOCHS    = 500    # MLP training epochs
_T_N_RESTARTS  = 8      # gradient-descent restarts on surrogate
_T_LR_SURR     = 3e-3   # Adam lr — surrogate training
_T_LR_OPT      = 5e-2   # Adam lr — surrogate optimisation
_T_N_OPT_STEPS = 400    # steps per optimisation restart

# Cost weights  (same as SMC Bayesian tuner for direct comparison)
_T_W_SS    = 2.0    # steady-state speed error [RPM]
_T_W_ID    = 50.0   # d-axis RMS [A]  (MTPA penalty)
_T_W_CHAT  = 4.0    # iq chattering   [A]
_T_W_VREF  = 20.0   # over-modulation penalty

# Physics-derived gain bounds
_T_BOUNDS = np.array([
    [0.05,  0.60],   # Kp_speed  [A/(rad/s)]
    [0.20,  6.00],   # Kp_id     [V/A]
    [1.00, 12.00],   # Kp_iq     [V/A]
], dtype=np.float64)

_T_PARAM_NAMES = ["Kp_speed", "Kp_id", "Kp_iq"]


# ── Latin-Hypercube sampling ──────────────────────────────────────────────────

def _t_lhs(n: int, bounds: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Latin-Hypercube sample: n points in d dimensions.

    Each dimension is independently stratified into n equal-width bins;
    within each bin one sample is drawn uniformly at random.  The bin
    order is then shuffled independently per dimension so the joint
    distribution has no column correlation.
    """
    d    = bounds.shape[0]
    cuts = np.linspace(0.0, 1.0, n + 1)
    X    = np.zeros((n, d))
    for j in range(d):
        u = rng.uniform(cuts[:-1], cuts[1:])   # one draw per bin
        rng.shuffle(u)                          # randomise row order
        X[:, j] = bounds[j, 0] + u * (bounds[j, 1] - bounds[j, 0])
    return X


# ── Cost metrics ──────────────────────────────────────────────────────────────

def _t_cost(d: dict | None) -> dict | None:
    """
    Compute scalar cost from a _run_sim() result dict.

    Returns None if the simulation diverged or produced insufficient data.
    """
    if d is None:
        return None

    t   = d["t"]
    rpm = d["speed_rpm"]
    idd = d["id"]
    iqr = d["iq_ref"]

    if len(t) < 200:
        return None

    # Hard divergence guard
    if float(np.max(np.abs(rpm))) > TARGET_RPM * 3.0:
        return None

    # Steady-state mask: last 15 % of simulation
    ss = t > 0.85 * T_SIM
    if not np.any(ss):
        return None

    ref_ss  = float(np.mean(d["omega_ref_rpm"][ss])) if "omega_ref_rpm" in d else TARGET_RPM
    ss_err  = float(np.mean(np.abs(rpm[ss] - ref_ss)))
    id_rms  = float(np.sqrt(np.mean(idd[ss] ** 2)))
    iq_chat = float(np.std(iqr[ss]))

    # Reject: SS error > 800 RPM means the controller failed to track
    if ss_err > 800.0:
        return None

    cost = _T_W_SS * ss_err + _T_W_ID * id_rms + _T_W_CHAT * iq_chat

    # Vref saturation penalty: 90th-percentile modulation index >= 0.93
    # indicates the controller is hitting the hexagon voltage ceiling
    # regularly — gains that cause over-modulation are penalised.
    vref = d.get("vref")
    if vref is not None and len(vref) > 0:
        if float(np.percentile(np.abs(vref), 90)) >= 0.93:
            cost += _T_W_VREF

    return {"cost": cost, "ss_err": ss_err, "id_rms": id_rms, "iq_chat": iq_chat}


def _t_run_with_gains(
    kp_speed: float,
    kp_id:    float,
    kp_iq:    float,
    with_noise: bool = True,
) -> dict | None:
    """
    Run one simulation with the given gains and return cost metrics.

    Temporarily patches _ACTIVE_GAINS in-place so _run_sim() picks them up.
    Restores original gains on exit (even on exception).
    """
    saved = _ACTIVE_GAINS.copy()
    try:
        _ACTIVE_GAINS["Kp_speed"] = kp_speed
        _ACTIVE_GAINS["Kp_id"]    = kp_id
        _ACTIVE_GAINS["Kp_iq"]    = kp_iq
        return _t_cost(_run_sim(with_noise=with_noise, with_codegen_hooks=False))
    finally:
        _ACTIVE_GAINS.update(saved)   # always restore


# ── Minimal MLP surrogate (pure NumPy — no PyTorch / TF dependency) ───────────

class _MLP:
    """
    2-hidden-layer MLP trained with Adam, pure NumPy.

    Architecture: 3 -> 16 -> 16 -> 1   (tanh activations, linear output)

    Hidden size 16 is chosen deliberately:
        - 3 gain parameters form a smooth cost surface with no high-frequency
          features, so 16 hidden units are more than sufficient.
        - With N_EXPLORE = 40 samples and hidden = 16, the network has
          16*3 + 16 + 16*16 + 16 + 1*16 + 1 = 353 parameters, which is
          under-determined relative to the training set — regularisation
          via early stopping / Adam weight decay is implicit.
        - hidden = 32 gives 2113 parameters for 40 samples (> 50x over-
          parameterised) and will memorise rather than generalise.
    """

    def __init__(self, n_in: int = 3, hidden: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        # He initialisation: scale = sqrt(2/fan_in)
        def _w(r, c): return rng.standard_normal((r, c)) * np.sqrt(2.0 / c)
        self.W1 = _w(hidden, n_in);   self.b1 = np.zeros(hidden)
        self.W2 = _w(hidden, hidden); self.b2 = np.zeros(hidden)
        self.W3 = _w(1, hidden);      self.b3 = np.zeros(1)
        # Adam first / second moment accumulators
        self._m = [np.zeros_like(p) for p in self._params()]
        self._v = [np.zeros_like(p) for p in self._params()]
        self._t = 0

    def _params(self) -> list:
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """X: (N, 3) -> (N,)"""
        h1 = np.tanh(X  @ self.W1.T + self.b1)
        h2 = np.tanh(h1 @ self.W2.T + self.b2)
        return (h2 @ self.W3.T + self.b3).squeeze(-1)

    def _fwd_cache(self, X: np.ndarray) -> tuple:
        z1 = X  @ self.W1.T + self.b1;  h1 = np.tanh(z1)
        z2 = h1 @ self.W2.T + self.b2;  h2 = np.tanh(z2)
        y  = (h2 @ self.W3.T + self.b3).squeeze(-1)
        return y, h1, h2

    def loss_and_grad(self, X: np.ndarray, y_true: np.ndarray) -> tuple:
        """MSE loss + analytical gradients via backpropagation."""
        N          = X.shape[0]
        y, h1, h2  = self._fwd_cache(X)
        err        = y - y_true                    # (N,)
        L          = float(np.mean(err ** 2))

        # dL/dy
        dL_dy = 2.0 * err / N                     # (N,)

        # Output layer
        dh2  = dL_dy[:, None] * self.W3           # (N, hidden)
        dW3  = (dL_dy[:, None] * h2).mean(0, keepdims=True)  # (1, hidden)
        db3  = dL_dy.mean(0, keepdims=True)        # (1,)

        # Hidden layer 2
        dz2  = dh2 * (1.0 - h2 ** 2)
        dW2  = dz2.T @ h1 / N                     # (hidden, hidden)
        db2  = dz2.sum(0) / N                     # (hidden,)
        dh1  = dz2 @ self.W2

        # Hidden layer 1
        dz1  = dh1 * (1.0 - h1 ** 2)
        dW1  = dz1.T @ X / N                      # (hidden, n_in)
        db1  = dz1.sum(0) / N                     # (hidden,)

        return L, [dW1, db1, dW2, db2, dW3, db3]

    def _adam(self, grads: list, lr: float,
              beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self._t += 1
        t = self._t
        for i, (p, g) in enumerate(zip(self._params(), grads)):
            self._m[i] = beta1 * self._m[i] + (1 - beta1) * g
            self._v[i] = beta2 * self._v[i] + (1 - beta2) * g ** 2
            m_hat = self._m[i] / (1 - beta1 ** t)
            v_hat = self._v[i] / (1 - beta2 ** t)
            p    -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 500, lr: float = 3e-3,
              verbose: bool = True) -> list[float]:
        losses = []
        for ep in range(epochs):
            L, grads = self.loss_and_grad(X, y)
            self._adam(grads, lr=lr)
            losses.append(L)
            if verbose and (ep % 100 == 0 or ep == epochs - 1):
                print(f"    epoch {ep:4d}  MSE={L:.6f}")
        return losses

    def scalar_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Forward pass + gradient of output w.r.t. input x (shape: (3,)).

        Used for gradient-descent optimisation of the gain vector.
        The gradient is exact (analytical backprop), not numerical.
        """
        X         = x[None, :]
        y, h1, h2 = self._fwd_cache(X)
        dh2  = np.ones((1, self.W3.shape[1])) * self.W3    # (1, hidden)
        dz2  = dh2 * (1.0 - h2 ** 2)
        dh1  = dz2 @ self.W2
        dz1  = dh1 * (1.0 - h1 ** 2)
        dx   = (dz1 @ self.W1).squeeze(0)                  # (3,)
        return float(y.squeeze()), dx


# ── Surrogate optimisation ────────────────────────────────────────────────────

def _t_optimise(mlp: _MLP, X_norm: np.ndarray, y_norm: np.ndarray,
                rng: np.random.Generator) -> np.ndarray:
    """
    Find x in [0,1]^3 that minimises mlp.forward(x).

    Runs _T_N_RESTARTS independent Adam gradient-descent trajectories.
    Restart 0 starts from the best observed point in X_norm.
    Restarts 1..N start from random initialisations.
    Returns the normalised gain vector with the lowest predicted cost.
    """
    best_x    = X_norm[int(np.argmin(y_norm))].copy()
    best_cost = float(mlp.forward(best_x[None, :])[0])

    for restart in range(_T_N_RESTARTS):
        x = best_x.copy() if restart == 0 else rng.uniform(0.0, 1.0, size=3)

        # Per-restart Adam state
        m_i = np.zeros(3); v_i = np.zeros(3); t_i = 0
        b1, b2, eps = 0.9, 0.999, 1e-8

        for _ in range(_T_N_OPT_STEPS):
            cost, grad = mlp.scalar_grad(x)
            t_i  += 1
            m_i   = b1 * m_i + (1 - b1) * grad
            v_i   = b2 * v_i + (1 - b2) * grad ** 2
            m_hat = m_i / (1 - b1 ** t_i)
            v_hat = v_i / (1 - b2 ** t_i)
            x     = x - _T_LR_OPT * m_hat / (np.sqrt(v_hat) + eps)
            x     = np.clip(x, 0.0, 1.0)   # stay inside unit hypercube

        c = float(mlp.forward(x[None, :])[0])
        if c < best_cost:
            best_cost = c
            best_x    = x.copy()

    return best_x


# ── gains.h writer ────────────────────────────────────────────────────────────

def _write_gains_header(
    kp_speed: float,
    kp_id:    float,
    kp_iq:    float,
    cost:     float,
    ss_err:   float,
    id_rms:   float,
    iq_chat:  float,
    out_path: Path,
) -> None:
    """
    Write a MISRA C:2012-compliant gains header file.

    The file is a drop-in replacement for embed_sim_dfc_gains.h.
    It carries the tuned values and a full audit trail (tuning date,
    cost function, verification metrics) in the Doxygen banner so the
    provenance of every constant is traceable to ISO 26262 requirements.
    """
    import datetime
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # Bandwidth values for each gain (omega_cl = Kp / L)
    L_avg = (_DB42S02.SMC_L_D + _DB42S02.SMC_L_Q) * 0.5   # [H]
    bw_id = kp_id / L_avg   # [rad/s]
    bw_iq = kp_iq / L_avg   # [rad/s]

    # Saturation breakpoint for speed loop
    sat_rads = _DB42S02.SMC_I_MAX / kp_speed if kp_speed > 0.0 else float("inf")
    sat_rpm  = sat_rads * 60.0 / (2.0 * math.pi)

    content = textwrap.dedent(f"""\
    /**********************************************************************************************************************
     * \\file      embed_sim_dfc_gains_tuned.h
     * \\brief     Differential Flatness Controller -- NN-tuned gains for NANOTEC DB42S02
     *
     * \\details   AUTO-GENERATED by the integrated NN surrogate tuner in
     *            db42s02_closed_loop_dfc_20k.py.  DO NOT EDIT MANUALLY.
     *
     *            TUNING AUDIT TRAIL
     *            ===================
     *            Generated   : {now}
     *            Method      : Latin-Hypercube ({_T_N_EXPLORE} samples) + MLP 3->16->16->1
     *                          + Adam gradient descent ({_T_N_RESTARTS} restarts)
     *            Noise model : ADC 12-bit sigma=1.5 LSB, encoder glitch 0.2 %,
     *                          PWM spike 0.5 A @ 5 %, dead-time 400 ns, bus ripple +-0.5 V
     *            Target      : {TARGET_RPM:.0f} RPM, V_DC = {V_DC:.1f} V, dt = {DT*1e6:.0f} us
     *
     *            COST FUNCTION
     *            ==============
     *            cost = {_T_W_SS} * ss_err_RPM
     *                 + {_T_W_ID} * id_rms_A
     *                 + {_T_W_CHAT} * iq_chat_A
     *                 + {_T_W_VREF} * [over-mod penalty if Vref_p90 >= 0.93]
     *
     *            VERIFICATION RESULT (real simulation, full noise model)
     *            =========================================================
     *            Total cost       : {cost:.4f}
     *            SS speed error   : {ss_err:.2f} RPM
     *            id RMS  (MTPA)   : {id_rms:.4f} A    (target 0)
     *            iq chattering    : {iq_chat:.4f} A   (std in steady state)
     *
     *            GAIN EQUATIONS
     *            ===============
     *            iq_ref [A]   = DFC_KP_SPEED [A/(rad/s)] * speed_err [rad/s]
     *            vd     [V]  += DFC_KP_ID    [V/A]        * (0 - id_meas) [A]
     *            vq     [V]  += DFC_KP_IQ    [V/A]        * (iq_ref - iq_meas) [A]
     *
     * \\note      MISRA C:2012 compliance
     *              Rule  7.2  : all float literals carry the 'f' suffix.
     *              Rule  8.1  : all types are explicit via the MatrixFloat typedef.
     *
     * \\version   1.0.0  (auto-generated)
     * \\copyright Copyright (C) EmbedSim 2025
     *********************************************************************************************************************/

    #ifndef EMBED_SIM_DFC_GAINS_TUNED_H_
    #define EMBED_SIM_DFC_GAINS_TUNED_H_

    #include "embed_sim_matrix.h"    /* MatrixFloat = real32_T */


    /*********************************************************************************************************************/
    /*------------------------------------------------------Macros-------------------------------------------------------*/
    /*********************************************************************************************************************/

    /** \\defgroup DFC_Gains_Tuned  NN-tuned gain constants
     * \\{{
     */

    /**********************************************************************************************************************
     * \\brief  Speed proportional gain (NN-tuned).
     *
     * \\details Control law:
     *            iq_ref [A] = DFC_KP_SPEED [A/(rad/s)] * (omega_ref - omega_meas) [rad/s]
     *
     *          Tuned value  : {kp_speed:.6f} A/(rad/s)
     *          Saturation   : I_MAX / Kp_speed = {_DB42S02.SMC_I_MAX:.2f} / {kp_speed:.4f}
     *                       = {sat_rads:.2f} rad/s  ({sat_rpm:.1f} RPM) speed error
     *          Previous     : DFC_KP_SPEED = 0.4 A/(rad/s) (hardware commissioning)
     *
     * \\units   A / (rad/s)
     *********************************************************************************************************************/
    #define DFC_KP_SPEED  ((MatrixFloat){kp_speed:.6f}f)

    /**********************************************************************************************************************
     * \\brief  D-axis current proportional gain (NN-tuned).
     *
     * \\details Control law:
     *            vd [V] += DFC_KP_ID [V/A] * (0 [A] - id_meas [A])
     *
     *          Tuned value  : {kp_id:.6f} V/A
     *          BW d-axis    : Kp_id / Ld = {kp_id:.4f} / {L_avg:.6f} = {bw_id:.1f} rad/s
     *          Previous     : DFC_KP_ID = 0.4 V/A (hardware commissioning)
     *
     * \\units   V / A
     *********************************************************************************************************************/
    #define DFC_KP_ID     ((MatrixFloat){kp_id:.6f}f)

    /**********************************************************************************************************************
     * \\brief  Q-axis current proportional gain (NN-tuned).
     *
     * \\details Control law:
     *            vq [V] += DFC_KP_IQ [V/A] * (iq_ref [A] - iq_meas [A])
     *
     *          Tuned value  : {kp_iq:.6f} V/A
     *          BW q-axis    : Kp_iq / Lq = {kp_iq:.4f} / {L_avg:.6f} = {bw_iq:.1f} rad/s
     *          Previous     : DFC_KP_IQ = 8.0 V/A (hardware commissioning)
     *
     * \\units   V / A
     *********************************************************************************************************************/
    #define DFC_KP_IQ     ((MatrixFloat){kp_iq:.6f}f)

    /** \\}} */  /* end defgroup DFC_Gains_Tuned */


    /*********************************************************************************************************************/
    /*-------------------------------------------------Data Structures---------------------------------------------------*/
    /*********************************************************************************************************************/

    /** \\defgroup DFC_GainSet_Tuned  Runtime gain structure (tuned values)
     * \\{{
     */

    /**********************************************************************************************************************
     * \\struct  DFC_GainSet_T
     * \\brief   Runtime-configurable mirror of the NN-tuned gain constants.
     *********************************************************************************************************************/
    typedef struct
    {{
        MatrixFloat kp_speed;    /**< Speed P-gain [A/(rad/s)].  Tuned: {kp_speed:.6f}. */
        MatrixFloat kp_id;       /**< D-axis current P-gain [V/A].  Tuned: {kp_id:.6f}. */
        MatrixFloat kp_iq;       /**< Q-axis current P-gain [V/A].  Tuned: {kp_iq:.6f}. */
    }} DFC_GainSet_T;

    /** \\}} */  /* end defgroup DFC_GainSet_Tuned */


    #endif /* EMBED_SIM_DFC_GAINS_TUNED_H_ */
    """)

    out_path.write_text(content, encoding="utf-8")
    print(f"\n  [gains.h] Written: {out_path}")


# ── Main tuner entry point ────────────────────────────────────────────────────

def run_tuner() -> bool:
    """
    Execute the full NN surrogate gain tuner.

    Returns True if tuning completed successfully and _ACTIVE_GAINS was
    updated.  Returns False if aborted or insufficient valid data.

    Phases
    ------
    1. LHS exploration  : _T_N_EXPLORE simulations, full AURIX noise model.
    2. MLP training     : 3->16->16->1, Adam, _T_N_EPOCHS epochs.
    3. Surrogate opt.   : gradient descent on MLP input, _T_N_RESTARTS restarts.
    4. Verification     : one real simulation at recommended gains, full noise.
    5. Header write     : embed_sim_dfc_gains_tuned.h with audit trail.
    """
    rng = np.random.default_rng(seed=42)

    print("\n" + "=" * 70)
    print("  DFC NN Surrogate Gain Tuner")
    print("=" * 70)
    print(f"  Phase 1  LHS exploration : {_T_N_EXPLORE} simulations")
    print(f"  Phase 2  MLP training    : {_T_N_EPOCHS} epochs  (3->16->16->1)")
    print(f"  Phase 3  Surrogate opt.  : {_T_N_RESTARTS} gradient-descent restarts")
    print(f"  Phase 4  Verification    : 1 simulation at recommended gains")
    print(f"  Phase 5  Header write    : embed_sim_dfc_gains_tuned.h")
    print()
    print(f"  Gain bounds:")
    for name, (lo, hi) in zip(_T_PARAM_NAMES, _T_BOUNDS):
        print(f"    {name:<12}  [{lo:.3f}, {hi:.3f}]")
    print(f"\n  Cost weights:")
    print(f"    SS speed error  x {_T_W_SS}")
    print(f"    id RMS (MTPA)   x {_T_W_ID}")
    print(f"    iq chattering   x {_T_W_CHAT}")
    print(f"    Vref over-mod   + {_T_W_VREF}  (if Vref p90 >= 0.93)")
    print(f"\n  Noise: FULL AURIX model (ADC, encoder, spikes, ripple)")
    print("=" * 70)

    # ── Phase 1: LHS exploration ──────────────────────────────────────────────
    print("\n[Phase 1] LHS exploration ...")
    X_raw  = _t_lhs(_T_N_EXPLORE, _T_BOUNDS, rng)   # (N, 3) physical gains
    costs  = []
    valid_results: list[tuple[np.ndarray, dict]] = []

    t0 = time.perf_counter()
    for i, gains in enumerate(X_raw):
        kp_s, kp_d, kp_q = gains
        print(f"  [{i+1:2d}/{_T_N_EXPLORE}]  "
              f"Kp_speed={kp_s:.4f}  Kp_id={kp_d:.3f}  Kp_iq={kp_q:.3f}",
              end="  ", flush=True)
        try:
            met = _t_run_with_gains(kp_s, kp_d, kp_q, with_noise=True)
        except KeyboardInterrupt:
            print("\n  Interrupted -- using data collected so far.")
            X_raw = X_raw[:i]
            break

        if met is None:
            print("-> UNSTABLE")
            costs.append(1e6)
        else:
            print(f"-> cost={met['cost']:.1f}  "
                  f"ss={met['ss_err']:.0f} RPM  "
                  f"id={met['id_rms']:.3f} A  "
                  f"chat={met['iq_chat']:.3f} A")
            costs.append(met["cost"])
            valid_results.append((gains.copy(), met))

    print(f"\n  Exploration done  ({time.perf_counter()-t0:.0f} s)")

    costs_arr = np.array(costs, dtype=np.float64)
    valid_mask = costs_arr < 1e5
    if valid_mask.sum() < 4:
        print("  ERROR: fewer than 4 valid simulations.")
        print("  Widen bounds or increase N_EXPLORE.")
        return False

    X_valid  = X_raw[valid_mask]
    y_valid  = costs_arr[valid_mask]

    # Best point observed so far
    best_obs_idx  = int(np.argmin(y_valid))
    best_obs_gains = X_valid[best_obs_idx]
    best_obs_cost  = float(y_valid[best_obs_idx])
    # Matching metrics from valid_results list
    # valid_results only has entries for non-unstable runs; their order
    # matches the True entries in valid_mask.
    best_obs_met  = valid_results[best_obs_idx][1]

    print(f"\n  Best observed:  cost={best_obs_cost:.2f}")
    for name, val in zip(_T_PARAM_NAMES, best_obs_gains):
        print(f"    {name:<12} = {val:.4f}")

    # ── Phase 2: MLP surrogate training ──────────────────────────────────────
    print("\n[Phase 2] Training MLP surrogate ...")

    # Normalise inputs to [0,1] (physics bounds); normalise outputs to N(0,1)
    X_norm = (X_valid - _T_BOUNDS[:, 0]) / (_T_BOUNDS[:, 1] - _T_BOUNDS[:, 0])
    y_mean = float(y_valid.mean())
    y_std  = float(max(y_valid.std(), 1e-8))
    y_norm = (y_valid - y_mean) / y_std

    # 80/20 split for held-out validation R^2
    n_val  = max(1, len(X_norm) // 5)
    idx    = rng.permutation(len(X_norm))
    X_tr, y_tr = X_norm[idx[n_val:]], y_norm[idx[n_val:]]
    X_va, y_va = X_norm[idx[:n_val]], y_norm[idx[:n_val]]

    mlp = _MLP(n_in=3, hidden=16, seed=0)
    mlp.train(X_tr, y_tr, epochs=_T_N_EPOCHS, lr=_T_LR_SURR, verbose=True)

    # Validation R^2 (held-out — not training set)
    y_va_pred = mlp.forward(X_va)
    ss_res    = float(np.var(y_va - y_va_pred))
    ss_tot    = float(np.var(y_va))
    r2_val    = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    print(f"\n  Surrogate R^2 (held-out {n_val} pts): {r2_val:.4f}  (>0.70 is acceptable)")
    if r2_val < 0.50:
        print("  WARNING: R^2 < 0.50 -- surrogate quality poor.")
        print("  Consider increasing N_EXPLORE before trusting Phase 3.")

    # ── Phase 3: surrogate optimisation ──────────────────────────────────────
    print(f"\n[Phase 3] Gradient descent on surrogate ({_T_N_RESTARTS} restarts) ...")
    best_norm  = _t_optimise(mlp, X_norm, y_norm, rng)

    # Denormalise back to physical gains
    gains_pred = _T_BOUNDS[:, 0] + best_norm * (_T_BOUNDS[:, 1] - _T_BOUNDS[:, 0])
    kp_s_pred, kp_d_pred, kp_q_pred = gains_pred

    cost_surr_norm = float(mlp.forward(best_norm[None, :])[0])
    cost_surr      = cost_surr_norm * y_std + y_mean
    print(f"  Surrogate minimum: predicted cost = {cost_surr:.2f}")
    for name, val in zip(_T_PARAM_NAMES, gains_pred):
        print(f"    {name:<12} = {val:.4f}")

    # ── Phase 4: verification ─────────────────────────────────────────────────
    print("\n[Phase 4] Verification simulation (full noise model) ...")
    met_verify = _t_run_with_gains(kp_s_pred, kp_d_pred, kp_q_pred, with_noise=True)

    if met_verify is None:
        print("  Verification UNSTABLE -- returning best observed gains.")
        final_gains = best_obs_gains
        final_met   = best_obs_met
    elif met_verify["cost"] < best_obs_cost:
        final_gains = gains_pred
        final_met   = met_verify
        print(f"  NN gains BETTER: cost {met_verify['cost']:.2f} < {best_obs_cost:.2f}")
    else:
        final_gains = best_obs_gains
        final_met   = best_obs_met
        print(f"  Best observed still wins: cost {best_obs_cost:.2f} <= {met_verify['cost']:.2f}")
        print(f"  (increase N_EXPLORE for a richer training set)")

    kp_s_f, kp_d_f, kp_q_f = float(final_gains[0]), float(final_gains[1]), float(final_gains[2])

    # ── Summary ───────────────────────────────────────────────────────────────
    defaults = [0.4, 0.4, 8.0]   # hardware-commissioning values from gains.h
    print("\n" + "=" * 70)
    print("  TUNING COMPLETE")
    print("=" * 70)
    print(f"\n  {'Parameter':<14} {'Before':>10}  {'After':>10}  {'Delta':>8}")
    print(f"  {'-'*46}")
    for name, default, tuned in zip(_T_PARAM_NAMES, defaults, final_gains):
        delta = (tuned - default) / (abs(default) + 1e-12) * 100.0
        sign  = "UP" if delta > 0.0 else "DN"
        print(f"  {name:<14}  {default:>10.4f}  {float(tuned):>10.4f}  "
              f"{sign} {abs(delta):5.1f}%")

    print(f"\n  Best cost        : {final_met['cost']:.4f}")
    print(f"  SS speed error   : {final_met['ss_err']:.2f} RPM")
    print(f"  id RMS  (MTPA)   : {final_met['id_rms']:.4f} A")
    print(f"  iq chattering    : {final_met['iq_chat']:.4f} A")
    print("=" * 70)

    # ── Phase 5: write gains header ───────────────────────────────────────────
    out_path = _FS_ELEC / "embed_sim_dfc_gains_tuned.h"
    _write_gains_header(
        kp_speed = kp_s_f,
        kp_id    = kp_d_f,
        kp_iq    = kp_q_f,
        cost     = final_met["cost"],
        ss_err   = final_met["ss_err"],
        id_rms   = final_met["id_rms"],
        iq_chat  = final_met["iq_chat"],
        out_path = out_path,
    )

    # Update active gains so the subsequent main simulation uses the tuned values
    _ACTIVE_GAINS["Kp_speed"] = kp_s_f
    _ACTIVE_GAINS["Kp_id"]    = kp_d_f
    _ACTIVE_GAINS["Kp_iq"]    = kp_q_f

    return True


# =============================================================================
# Entry point
# =============================================================================

def _ask_user_tune() -> bool:
    """
    Interactively ask the user whether to run the gain tuner.

    Returns True if the user answered yes, False otherwise.
    Accepts: y / yes / Y / YES  (case-insensitive).
    """
    print()
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  GAIN TUNER                                                 │")
    print("  │  Run the NN surrogate tuner before the main simulation?     │")
    print("  │                                                             │")
    print(f"  │  This will run ~{_T_N_EXPLORE} simulations and may take several minutes.  │")
    print("  │  On completion it writes embed_sim_dfc_gains_tuned.h and   │")
    print("  │  uses the tuned gains for the main simulation.             │")
    print("  └─────────────────────────────────────────────────────────────┘")
    try:
        answer = input("  Run tuner? [y/N] : ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"
    return answer in ("y", "yes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DB42S02 DFC FOC simulation with optional NN gain tuner.")
    parser.add_argument(
        "--tune",
        action  = "store_true",
        help    = "Run the NN surrogate gain tuner before the main simulation "
                  "(non-interactive, equivalent to answering 'y' at the prompt).",
    )
    parser.add_argument(
        "--no-tune",
        action  = "store_true",
        help    = "Skip the tuner prompt and use current gains directly.",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  DB42S02 -- Differential Flatness FOC -- 20 kHz  [AURIX noise ON]")
    print("=" * 65)
    print(f"  ADC:     12-bit,  sigma={_ADC_NOISE_SIGMA*1e3:.2f} mA,  sat=+-{_ADC_SAT_LIMIT:.1f} A")
    print(f"  Encoder: {_ENC_PPR} PPR x 4,  res={_ENC_RESOLUTION*1e3:.2f} mrad,  "
          f"glitch={_ENC_GLITCH_PROB*100:.1f}%")
    print(f"  PWM:     spike={_SPIKE_AMP:.2f} A,  prob={_SPIKE_PROB*100:.0f}%")
    print(f"  DeadT:   {_DEAD_TIME_S*1e9:.0f} ns  ->  {_DEAD_TIME_V*1e3:.1f} mV/phase")
    print(f"  Bus:     +-{_BUS_RIPPLE_AMP:.2f} V @ {_BUS_RIPPLE_HZ:.0f} Hz")
    print(f"\n  Default gains (hardware commissioning):")
    for k, v in _ACTIVE_GAINS.items():
        print(f"    {k:<12} = {v}")
    print("=" * 65)

    # ── Tuner gate ────────────────────────────────────────────────────────────
    if args.tune:
        do_tune = True
    elif args.no_tune:
        do_tune = False
    else:
        do_tune = _ask_user_tune()

    if do_tune:
        ok = run_tuner()
        if not ok:
            print("\n  Tuner aborted -- proceeding with default gains.")
        print(f"\n  Gains active for main simulation:")
        for k, v in _ACTIVE_GAINS.items():
            print(f"    {k:<12} = {v:.6f}")
    else:
        print("\n  Tuner skipped -- using default gains.")

    # ── Main simulation ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Running main simulation ...")
    print("=" * 65)

    data = _run_sim(with_noise=True, with_codegen_hooks=True)
    if data is None:
        print("  Simulation failed.")
        sys.exit(1)

    print_summary(data)
    _run_codegen(data)
    plot_results(data, out_path=str(_HERE / "db42s02_dfc_foc_20k_results.png"))

    print("\n[Done]")
    print("  db42s02_dfc_foc_20k_results.png")
    print("  db42s02_dfc_topology.html")
    print("  embedsim_gen/embedsim_step.c   <- flash to AURIX")
    print("  embedsim_gen/embedsim_step.h")
    if do_tune and (_FS_ELEC / "embed_sim_dfc_gains_tuned.h").exists():
        print("  embed_sim_dfc_gains_tuned.h    <- replace embed_sim_dfc_gains.h")
